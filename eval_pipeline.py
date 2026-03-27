#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import re
import json
import torch
import argparse
from tqdm import tqdm
import tempfile
import multiprocessing as mp
from transformers import logging
from collections import defaultdict
logging.set_verbosity_error()

try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            # best-effort fallback; mirrors typical semantics
            return self.get_seq_length()
        DynamicCache.get_max_length = _get_max_length
except Exception:
    pass

from pipeline import get_pipeline
from info import VIDEO_ROOT
from prompt import EVAL_PROMPT

def _eval_subset(args, qa_subset, pipeline, proc_idx, gpu_id, tmp_path):
    """Evaluate a subset of qa_pairs on one process/GPU and write results to tmp_path."""
    responses = []
    yes_cnt = 0
    correct = 0
    invalid_cnt = 0
    total = len(qa_subset)

    desc = f"[GPU {gpu_id} | Proc {proc_idx}]"
    for qa_pair in tqdm(qa_subset, desc=desc):
        video_path = qa_pair['video_path']
        video_id = video_path.split("test_eval/")[-1]
        video_path = os.path.join(VIDEO_ROOT, video_id)
        assert os.path.exists(video_path), f"Video path {video_path} does not exist."

        question_type = qa_pair['question_type']
        dimension = qa_pair['dimension']
        question = qa_pair['question']
        gt_answer = qa_pair['answer']

        prompt = EVAL_PROMPT.format(question=question, question_type=question_type)
        response = pipeline.generate_response(prompt, video_path, video_id, q_type=question_type)

        if isinstance(response, str):
            curr_response = re.sub(r"```json|```", "", response).strip()
            try:
                curr_response = json.loads(curr_response)
                model_answer = curr_response.get('answer', "").strip().lower()
            except json.JSONDecodeError:
                print(f"[{desc}] Invalid response for video {video_id}: {curr_response}")
                model_answer = "ERROR"
                invalid_cnt += 1
        elif isinstance(response, dict):
            model_answer = response.get('answer', "").strip().lower()
        elif response is None:
            model_answer = "ERROR"
            invalid_cnt += 1
        else:
            model_answer = str(response).strip().lower()

        is_correct = int(gt_answer.lower() == model_answer)
        correct += is_correct
        if 'yes' in model_answer:
            yes_cnt += 1

        responses.append({
            **qa_pair,
            "model_response": response,
            "correct": is_correct,
            "prompt": prompt,
        })

    # Persist this shard’s results
    shard = {
        "proc_idx": proc_idx,
        "gpu_id": gpu_id,
        "totals": {
            "total": total,
            "yes_cnt": yes_cnt,
            "correct": correct,
            "invalid_cnt": invalid_cnt
        },
        "responses": responses
    }
    with open(tmp_path, "w") as f:
        json.dump(shard, f, indent=2, ensure_ascii=False)

def _worker(proc_idx, qa_subset, num_gpus, args, tmp_dir):
    gpu_id = proc_idx % max(1, num_gpus)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
    print(device)
    # Build a fresh pipeline per process (important with spawn + CUDA)
    pipeline = get_pipeline(args, device)

    tmp_path = os.path.join(tmp_dir, f"shard_{proc_idx}.json")
    try:
        _eval_subset(args, qa_subset, pipeline, proc_idx, gpu_id, tmp_path)
    except Exception as e:
        # If something goes wrong, write an empty shard with the error noted
        err_payload = {
            "proc_idx": proc_idx,
            "gpu_id": gpu_id,
            "error": str(e),
            "totals": {"total": len(qa_subset), "yes_cnt": 0, "correct": 0, "invalid_cnt": len(qa_subset)},
            "responses": []
        }
        with open(tmp_path, "w") as f:
            json.dump(err_payload, f, indent=2, ensure_ascii=False)
        print(f"[GPU {gpu_id} | Proc {proc_idx}] ERROR: {e}")

def parallel_eval(args, qa_pairs):
    """
    Multi-GPU multi-processing eval.
    args must provide:
      - args.num_gpus (int)
      - args.procs_per_gpu (int)
      - args.save_dir (str)
      - args.model_name (str)
    """
    os.makedirs(args.save_dir, exist_ok=True)

    total_procs = max(1, args.num_gpus * args.procs_per_gpu)
    print(f"[INFO] Found {len(qa_pairs)} QA pairs. Spawning {total_procs} procs over {args.num_gpus} GPUs...")

    # Round-robin split of qa_pairs into shards
    shards = [[] for _ in range(total_procs)]
    for i, item in enumerate(qa_pairs):
        shards[i % total_procs].append(item)

    with tempfile.TemporaryDirectory() as tmp_dir:
        processes = []
        for proc_idx in range(total_procs):
            p = mp.Process(
                target=_worker,
                args=(proc_idx, shards[proc_idx], args.num_gpus, args, tmp_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge results
        all_responses = []
        agg_correct = agg_invalid = agg_total = 0

        # Collect shards in index order
        for proc_idx in range(total_procs):
            shard_path = os.path.join(tmp_dir, f"shard_{proc_idx}.json")
            if not os.path.exists(shard_path):
                print(f"[WARN] Missing shard file: {shard_path}")
                continue
            with open(shard_path, "r") as f:
                data = json.load(f)

            t = data.get("totals", {})
            agg_total += t.get("total", 0)
            agg_correct += t.get("correct", 0)
            agg_invalid += t.get("invalid_cnt", 0)
            all_responses.extend(data.get("responses", []))

    # Final stats + save combined file
    correct_ratio = (agg_correct / agg_total) if agg_total else 0.0

    print(f"correct ratio: {agg_correct} / {agg_total} = {correct_ratio:.3%}")

    save_path = os.path.join(args.save_dir, f"{args.model_name}-eval.json")
    with open(save_path, "w") as f:
        json.dump(all_responses, f, indent=4, ensure_ascii=False)
    print(f"Saved evaluation results to {save_path}")

def eval(args, qa_pairs, pipeline):
    responses = []
    yes_cnt = 0
    correct = 0
    total = len(qa_pairs)
    invalid_cnt = 0
    for qa_pair in tqdm(qa_pairs):
        video_path = qa_pair['video_path']
        video_id = video_path.split("test_eval/")[-1]
        assert os.path.exists(video_path), f"Video path {video_path} does not exist."

        question_type = qa_pair['question_type']
        dimension = qa_pair['dimension']
        question = qa_pair['question']
        gt_answer = qa_pair['answer']

        prompt = EVAL_PROMPT.format(question=question, question_type=question_type)
        response = pipeline.generate_response(prompt, video_path, video_id, q_type=question_type)
    
        if isinstance(response, str):
            curr_response = re.sub(r"```json|```", "", response).strip()
            try:
                curr_response = json.loads(curr_response)
                model_answer = curr_response['answer'].strip().lower()
            except json.JSONDecodeError:
                print(f"Invalid response for video {video_id}: {curr_response}")
                model_answer = "ERROR"
                invalid_cnt += 1
        elif isinstance(response, dict):
            model_answer = response['answer'].strip().lower()
        elif response is None:
            model_answer = "ERROR"
            invalid_cnt += 1

        correct += int(gt_answer.lower() == model_answer)
        yes_cnt += 1 if 'yes' in model_answer.lower() else 0
        responses.append({**qa_pair, "model_response": response, "correct": int(gt_answer.lower() == model_answer), 'prompt': prompt})
    print(f"correct ratio: {correct} / {total} = {correct / total:.3%}")
    
    save_path = os.path.join(args.save_dir, f"{args.model_name}-eval.json")
    with open(save_path, 'w') as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)
    print(f"Saved evaluation results to {save_path}")


def get_accuracy(file_path):
    with open(file_path, "r") as f: 
        responses = json.load(f)

    filtered_cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    filtered_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    total = 0
    total_correct = 0
    new_responses = []
    for response in responses:
        track_type = response['track_type']
        qtype = response['question_type']
        dimension = response['dimension']
        template_type = response['template_type']
        if template_type == "later_to_start_wo_scene": template_type = "later_to_start"
        elif template_type == "start_to_later_wo_scene": template_type = "start_to_later"
        total += 1
        filtered_cnt[track_type][dimension][qtype][template_type] += 1
        gt_answer = response['answer'].lower() if 'answer' in response else response['gt_answer'].lower()
        try:
            model_response = response['model_response'] if 'model_response' in response else response['model_answer']
            if isinstance(model_response, str):
                model_response = re.sub(r"```json|```", "", model_response).strip()
                model_answer = json.loads(model_response)['answer'].lower()
            elif isinstance(model_response, dict):
                model_answer = model_response['answer'].lower()
        except:
            if "{" not in model_response:
                model_answer = model_response.lower()
            elif not model_response.startswith("{"):
                try:
                    model_answer = json.loads(model_response.split("\n\n")[-1])['answer'].lower()
                except:
                    match = re.search(r'"answer"\s*:\s*"([^"]+)"', model_response)
                    model_answer = match.group(1).strip().lower() if match else "ERROR"
            else:
                model_answer = model_response.split('"answer": ')[-1].split(",")[0].strip('"').lower()
        
        if qtype == "ordering":
            gt_answer_set = gt_answer.split(",")
            model_answer_set = model_answer.split(",")
            if len(model_answer_set) == 1: correct = 0
            else:
                if gt_answer_set[0].strip().lower() == model_answer_set[0].strip().lower() and gt_answer_set[1].strip().lower() == model_answer_set[1].strip().lower():
                    correct = 1
                else:
                    correct = 0
        else:
            correct = int(gt_answer in model_answer)
        print(f"GT answer: {gt_answer}, Model answer: {model_answer}, correct: {correct}")
        new_responses.append({**response, "model_answer": model_answer})
        new_responses[-1]['correct'] = correct
        filtered_acc[track_type][dimension][qtype][template_type] += correct
        total_correct += correct

    with open(file_path.replace(".json", "_acc.json"), "w") as f:
        json.dump(new_responses, f, indent=4, ensure_ascii=False)
    print(f"Total Accuracy: {total_correct / total:.3%}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Pipeline")
    parser.add_argument("--task", type=str, choices=["eval", "get_acc"], default="eval")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use. Default: all available.")
    parser.add_argument("--procs_per_gpu", type=int, default=2, help="Number of GPUs to use. Default: all available.")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")

    parser.add_argument("--save_dir", type=str, default="eval_results")
    parser.add_argument("--qa_path", type=str, default=None)
    parser.add_argument("--resp_path", type=str, default=None)
    # open-source model parameters
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--save_kframe", action="store_true", help="Save key frames during processing")
    parser.add_argument("--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2-7B")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")

    # LLaVA-NeXT-Video parameters
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--mm_pooling_position", type=str, default="after")
    
    # VideoChat2 parameters
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--use_fps", action="store_true")
    args = parser.parse_args()

    if args.task == "eval":
        os.makedirs(args.save_dir, exist_ok=True)
        
        with open(args.qa_path, "r") as f:
            qa_pairs = json.load(f)

        if args.num_gpus == 1 and args.procs_per_gpu == 1:
            pipeline = get_pipeline(args, schema_type="eval")
            eval(args, qa_pairs, pipeline)
        else:
            mp.set_start_method('spawn', force=True)
            parallel_eval(args, qa_pairs)

    elif args.task == "get_acc":
        get_accuracy(args.resp_path)