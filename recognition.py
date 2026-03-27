#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
import re
from collections import defaultdict
from tqdm import tqdm
import argparse
import shutil
from collections import defaultdict

from pipeline.gemini import GeminiVideoPipeline
from prompt import GEMINI_ACTION_RECOG_PROMPT
from video_utils import extract_video_timestamps

class GeminiActionRecognizer:
    def __init__(self, args, prompt_template, action_prompt_template):
        self.args = args
        self.single_engine = GeminiVideoPipeline(args.gemini_model_name, schema_type="single")
        self.entire_engine = GeminiVideoPipeline(args.gemini_model_name, schema_type="entire")
        self.prompt = prompt_template
        self.action_prompt = action_prompt_template
        self.video_root = args.video_root
        self.result, self.result_entire = {}, {}

    def get_metadata_paths(self):
        target_filenames = {"appear_metadata.json", "disappear_metadata.json", "reappear_metadata.json"}
        json_paths = []
        for root, _, files in os.walk(self.video_root):
            for fname in files:
                if fname in target_filenames:
                    json_paths.append(os.path.join(root, fname))
        return json_paths

    def reconstruct_metadata(self, meta_info):
        new_metadata = []
        for item in meta_info:
            entity_tracking = item.get("entity_tracking", [])
            entity_info = []
            for tracking in entity_tracking:
                entity_info.append({
                    "name": tracking["name"],
                })
            new_metadata.append({
                "timestamp": item["timestamp"],
                "scene": item["scene"][0][0],
                "entity_info": entity_info
            })
        return new_metadata

    def process_file_as_a_chunk(self, metadata_path):
        # Extract actions
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        video_id = metadata_path.split("/")[-2]
        track_type = metadata_path.split("/")[-1].split("_")[0]
        save_path = metadata_path.replace("metadata", "metadata_gemini_single")
        if os.path.exists(save_path):
            print("[INFO] Metadata already processed, loading from cache...")
            with open(save_path, "r") as f:
                final_metadata = json.load(f)
            return final_metadata, video_id, track_type, save_path

        final_metadata = defaultdict(dict)
        
        for entity_id, candidates_dict in metadata.items():
            for time_range, meta_info in candidates_dict.items():
                print(f"[INFO] Processing {video_id} - {entity_id} - {track_type} - {time_range}")
                start_time, end_time = time_range.split("-")
                video_path = meta_info["video_path"].replace(".mp4", "_overlay.mp4")
                video_segments = meta_info["video_segments"]
                curr_metadata = self.reconstruct_metadata(meta_info["metadata"])
                ann_id = f"{video_id}_{entity_id}_{track_type}_{start_time}_{end_time}"

                tublets = []
                for segment in video_segments:
                    seg_start, seg_end = segment
                    out_dir = video_path.replace(".mp4", "")
                    curr_video_path = extract_video_timestamps(
                        video_path, 
                        out_dir=out_dir,
                        start_time=seg_start, 
                        end_time=seg_end, 
                        offset=0
                    )
                    response = self.single_engine.generate_response(self.action_prompt, curr_video_path, ann_id)
                    print(f"[INFO] Gemini response for {seg_start}: {response}")
                    if response is None or response['action'].lower() == "invalid": continue
                    tublets.append({
                        "start_time": seg_start,
                        "end_time": seg_end,
                        "action": response['action'],
                        "outfit": response['outfit'],
                        "scene": response['scene'],
                    })
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)

                if len(tublets) > 0:
                    self.result[ann_id] = {
                        "video_path": meta_info["video_path"],
                        "prompt": self.prompt,
                        "response": tublets
                    }

                    final_metadata[entity_id][f"{start_time}-{end_time}"] = {
                        "video_path": meta_info["video_path"],
                        "gemini_metadata": tublets,
                        "metadata": curr_metadata
                    }

        with open(save_path, "w") as f:
            json.dump(final_metadata, f, indent=4)
        print(f"[INFO] Saved Gemini metadata to {save_path}")
        return final_metadata, video_id, track_type, save_path

    def process_file_as_a_whole(self, video_id, track_type, final_metadata, save_path):
        print("=============== Processing as a whole ===============")
        new_final_metadata = defaultdict(dict)

        for entity_id, chunk_dict in final_metadata.items():
            for time_range, chunk_info in chunk_dict.items():
                start_time, end_time = time_range.split("-")
                ann_id = f"{video_id}_{entity_id}_{track_type}_{start_time}_{end_time}"

                video_path = chunk_info["video_path"].replace(".mp4", "_overlay.mp4")
                gemini_metadata = chunk_info['gemini_metadata']
                action_changes = [metadata['action'] for metadata in gemini_metadata if 'action' in metadata]
                scene_changes = [metadata['scene'] for metadata in gemini_metadata if 'scene' in metadata]
                outfit_changes = [metadata['outfit'] for metadata in gemini_metadata if 'outfit' in metadata]
                prompt = self.prompt.format(action_changes=action_changes, scene_changes=scene_changes, outfit_changes=outfit_changes)
                response = self.entire_engine.generate_response(
                    prompt,
                    video_path,
                    ann_id
                )
                
                if isinstance(response, str):
                    response = json.loads(re.sub(r"```json|```", "", response).strip())

                self.result_entire[ann_id] = {
                    "video_path": chunk_info["video_path"],
                    "prompt": prompt,
                    "response": response
                }

                new_final_metadata[entity_id][f"{start_time}-{end_time}"] = {
                    "video_path": chunk_info["video_path"],
                    "status_changes": {
                        "action_changes": action_changes,
                        "scene_changes": scene_changes,
                        "outfit_changes": outfit_changes
                    },
                    "gemini_metadata": response,
                    "metadata": chunk_info['metadata']
                }

        save_path = save_path.replace("metadata_gemini_single", "metadata_gemini")
        with open(save_path, "w") as f:
            json.dump(new_final_metadata, f, indent=4)
        print(f"[INFO] Saved Gemini metadata to {save_path}")

    def run(self):
        metadata_paths = self.get_metadata_paths()
        for path in tqdm(metadata_paths, desc="Processing metadata files"):
            print(f"[INFO] Processing {path}")
            final_metadata, video_id, track_type, save_path = self.process_file_as_a_chunk(path)
            self.process_file_as_a_whole(video_id, track_type, final_metadata, save_path)

        with open(os.path.join(self.args.video_root, "gemini_single_action_recog_result.json"), "w") as f:
            json.dump(self.result, f, indent=4)
        with open(os.path.join(self.args.video_root, "gemini_entire_action_recog_result.json"), "w") as f:
            json.dump(self.result_entire, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_model_name", type=str, default="gemini-2.0-flash", choices=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])
    parser.add_argument("--outfit_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--video_root", type=str, default="narrahalluc_videos")
    parser.add_argument("--result_path", type=str, default="gemini_action_recognition_result.json")
    args = parser.parse_args()

    action_recognizer = GeminiActionRecognizer(
        args,
        prompt_template=GEMINI_ACTION_RECOG_PROMPT
    )
    action_recognizer.run(result_output_path=os.path.join(args.video_root, args.result_path))
