#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
import json
import math
import re
import os
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def crop_outfit_region(frame, person_box, face_box):
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = face_box

    outfit_crop = frame[py1:py2, px1:px2].copy()
    rel_fx1, rel_fy1 = fx1 - px1, fy1 - py1
    rel_fx2, rel_fy2 = fx2 - px1, fy2 - py1
    cv2.rectangle(outfit_crop, (rel_fx1, rel_fy1), (rel_fx2, rel_fy2), (0, 0, 0), thickness=-1)
    return outfit_crop, [rel_fx1, rel_fy1, rel_fx2, rel_fy2]

def save_outfit_crop(out_dir, name, frame_idx, box_id, crop_img):
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    save_path = os.path.join(out_dir, name, f"frame_{frame_idx:06d}_box{box_id:03d}.jpg")
    cv2.imwrite(save_path, crop_img)
    return save_path

def replace_inner_quotes_in_thoughts(broken_json):
    match = re.search(r'("Thoughts":\s*")(.+)', broken_json, re.DOTALL)
    if not match:
        raise ValueError("No 'Thoughts' field found.")

    prefix = broken_json[:match.start(2)]
    thoughts_value = match.group(2)
    repaired_thoughts = thoughts_value.replace('"', "'")

    if not repaired_thoughts.endswith('"'):
        repaired_thoughts += '"'
    if not repaired_thoughts.endswith('"}'):
        repaired_thoughts += '}'

    repaired_json_str = prefix + repaired_thoughts
    return repaired_json_str

def replacer(match):
    key = match.group(1)
    value = match.group(2).strip()

    # Don't quote valid types
    if value in ['true', 'false', 'null']:
        return f'{key}: {value}'
    if re.fullmatch(r'-?\d+(\.\d+)?', value):
        return f'{key}: {value}'

    # Otherwise, wrap in quotes
    return f'{key}: "{value}"'

def fix_values_using_next_key_boundary(s):
    # Normalize whitespace
    s = ' '.join(s.strip().split())

    # Ensure all keys are quoted
    s = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', s)

    # Fix unquoted values using improved pattern
    pattern = r'(".*?")\s*:\s*([^"{\[\]]+?)(?=,\s*\"|\s*})'
    s = re.sub(pattern, replacer, s)
    return s

def load_jsonstr(target):
    try:
        result = json.loads(target) # '{\n    "Decision": "No", \n    "Thoughts": "The video does not contain multiple people or no one undergoes a noticeable action or state change. It only shows a person\'s hand holding a pen and a laptop on a table with a notebook and a pack of cigarettes."\n}'
    except:
        if target.endswith('"'): fixed = target + '\n}'
        else: fixed = target + '"\n}'

        try:
            result = json.loads(fixed)
        except:
            if target.startswith('{'):
                fixed = fix_values_using_next_key_boundary(target)
            else:
                fixed = target.split("{\n")[1]
                fixed = "{\n" + fixed
                fixed = fix_values_using_next_key_boundary(fixed)
            # fixed = replace_inner_quotes_in_thoughts(target)

            try:
                result = json.loads(fixed)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None
    return result

def convert_numpy(obj):
    """
    Recursively convert NumPy and PyTorch types into native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {
            str(convert_numpy(k)) if isinstance(k, (np.integer, np.floating)) else convert_numpy(k):
            convert_numpy(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj

def draw_distribution(values, save_path):
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    # 1-sigma range (within one standard deviation from mean)
    one_sigma_range = (mean - std, mean + std)

    # Quartiles
    q1 = np.percentile(arr, 25)
    q2 = np.percentile(arr, 50)  # median
    q3 = np.percentile(arr, 75)

    # Print summary
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print(f"1-Sigma Range: {one_sigma_range[0]:.2f} to {one_sigma_range[1]:.2f}")
    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Q2 (Median): {q2:.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")

    # Plot distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(arr, kde=True, bins=10)  # change to plt.hist(...) if not using seaborn
    plt.axvline(mean, color='r', linestyle='--', label='Mean')
    plt.axvline(one_sigma_range[0], color='g', linestyle=':', label='-1σ')
    plt.axvline(one_sigma_range[1], color='g', linestyle=':', label='+1σ')
    plt.axvline(q1, color='orange', linestyle='--', label='Q1')
    plt.axvline(q2, color='purple', linestyle='--', label='Median')
    plt.axvline(q3, color='orange', linestyle='--', label='Q3')
    plt.legend()
    plt.title("Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)

def get_subdirectories(path):
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

def get_valid_avg_setmiou():
    def get_valid_files_final(valid_files, avg_accs, thres):
        sets = set()
        valid_files_final = []
        for file, acc in zip(valid_files, avg_accs):
            if acc >= thres:
                print(f"{file}: {acc:.4f}")
                valid_files_final.append(file)
                sets.add(file.split("/")[-3])
        print(sets)
        return valid_files_final, sets

    base_dir = "PATH_TO_BASEDIR"
    valid_files = []
    avg_accs = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "metadata.json":
                print("Processing file:", os.path.join(root, file))
                ensemble_accs = []
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    for item in data:
                        if item["entity_detection"]["set_miou"] != 0 and not math.isnan(item["entity_detection"]["set_miou"]):
                            ensemble_accs.append(item["entity_detection"]["set_miou"])

                    if ensemble_accs:
                        avg_accs.append(np.mean(ensemble_accs))
                        valid_files.append(os.path.join(root, file))

                        with open(os.path.join(root, "detection_bbox_acc.json"), "r") as f:
                            detection_metadata = json.load(f)
                        
                        detection_metadata['average_set_mIoU_valid'] = np.mean(ensemble_accs)
                        with open(os.path.join(root, "detection_bbox_acc.json"), "w") as f:
                            json.dump(detection_metadata, f, indent=2)
                        print(f"Processing {os.path.join(root, file)}: Average set mIoU = {np.mean(ensemble_accs):.4f}")
    draw_distribution(avg_accs, os.path.join(base_dir, "lvbench_set_miou_distribution.png"))
    print(f"Total valid files: {len(valid_files)}")
    print(f"Average set mIoUs: {avg_accs}")
    entity_total = 0
    valid_video_total = 0
    valid_video_set = set()
    valid_files_final, sets = get_valid_files_final(valid_files, avg_accs, 0.47)
    for valid_file in valid_files_final:
        video_id = "/".join(valid_file.split("/")[-3:-1])
        face_dir = os.path.join(base_dir, video_id, "face_recog")
        if not os.path.exists(face_dir):
            print(f"Face directory does not exist: {face_dir}")
            continue

        subdirs = get_subdirectories(face_dir)
        entity_tmp = 0
        for subdir in subdirs:
            if 'removed' in subdir: continue
            entity_tmp += 1

        if entity_tmp > 0:
            valid_video_total += 1
            entity_total += entity_tmp
            valid_video_set.add(video_id.split("/")[-1])
        
    print(f"Total valid videos: {valid_video_total}, Total entities: {entity_total}")
    print(f"Valid video set: {valid_video_set}")

def check_qa_distr():
    qa_type_cnt = defaultdict(int)
    qa_subtype_cnt = defaultdict(int)
    answer_type_cnt = defaultdict(int)
    
    qa_path = "PATH_TO_QAFILE"
    with open(qa_path, "r") as f:
        qa_file = json.load(f)

    for item in qa_file:
        question_type = item['question_type']
        subtype = item['dimension']
        answer = item['answer']

        qa_type_cnt[question_type] += 1
        qa_subtype_cnt[subtype] += 1
        answer_type_cnt[answer] += 1

    print(f"Question Type Counts: {dict(qa_type_cnt)}")
    print(f"Question Subtype Counts: {dict(qa_subtype_cnt)}")
    print(f"Answer Type Counts: {dict(answer_type_cnt)}")

