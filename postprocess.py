#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
import csv
from collections import defaultdict
import pandas as pd
import argparse
from pathlib import Path

from metrics import percentage_agreement, compute_fleiss_kappa
from gemini_annotation import GeminiAnnotation

class PostProcess:
    def __init__(self, args, filtered_video_ids=None):
        self.args = args
        self.data_dir = args.data_dir
        self.lifespan_thres = args.lifespan_thres
        self.consec_dur_thres = args.consec_dur_thres
        self.frame_rate = args.frame_rate
        self.filtered_video_ids = filtered_video_ids

    def compute_consecutive_durations(self, timestamps):
        if not timestamps:
            return []
        timestamps = sorted(timestamps)
        durations = []
        start = prev = timestamps[0]
        for t in timestamps[1:]:
            if t == prev + 1:
                prev = t
            else:
                durations.append(prev - start + 1)
                start = prev = t
        durations.append(prev - start + 1)
        return durations

    def apply_lifespan_entity_tracking(self):
        removed = {}
        filtered_metadata_all = {}
        video_cnt = 0
        print(f"[INFO] Applying lifespan entity tracking: {self.data_dir}")
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file == "entity_tracking_metadata.json":
                    if self.args.video_type == "AVA":
                        video_id = os.path.basename(root)
                    else:
                        video_id = "/".join(root.split("/")[-2:])

                    if video_id not in self.filtered_video_ids: continue
                    video_cnt += 1

                    meta_path = os.path.join(root, "entity_tracking_metadata.json")
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)

                    print(f"[INFO] Processing {video_id}")
                    for entity in list(metadata.keys()):
                        if entity == "Unknown":
                            continue
                        entries = metadata[entity]
                        timestamps = [entry["timestamp"] for entry in entries]
                        if not timestamps:
                            continue
                        lifespan = max(timestamps) - min(timestamps) + 1
                        durations = self.compute_consecutive_durations(timestamps)
                        has_valid_segment = any(d >= self.consec_dur_thres for d in durations)

                        if lifespan < self.lifespan_thres or not has_valid_segment:
                            removed.setdefault(video_id, []).append({
                                "entity": entity,
                                "lifespan": lifespan,
                                "has_valid_segment": has_valid_segment
                            })
                            metadata.setdefault("Unknown", []).extend(entries)
                            del metadata[entity]
                            src_path = os.path.join(root, "face_recog", entity)
                            dst_path = os.path.join(root, "face_recog", f"removed_{entity}")
                            if not os.path.exists(src_path) and os.path.exists(dst_path):
                                print("[INFO] Already removed, skipping rename.")
                            else:
                                os.rename(src_path, dst_path)   
                                print(f"[INFO] Removed and renamed: {entity} → removed_{entity}")
                    
                    metadata["Unknown"].sort(key=lambda x: x["timestamp"])
                    save_path = os.path.join(root, "entity_tracking_metadata_filtered.json")
                    with open(save_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    filtered_metadata_all[video_id] = {
                        'num_valid_entities': len(metadata) - 1,
                        'removed_entities': removed.get(video_id, [])
                    }

        print(f"[INFO] Processed {video_cnt} videos, removed entities: {len(removed)}")
        with open(os.path.join(self.data_dir, "removed_entities.json"), "w") as f:
            json.dump(removed, f, indent=4)
        with open(os.path.join(self.data_dir, "filtered_metadata_all.json"), "w") as f:
            json.dump(filtered_metadata_all, f, indent=4)
        return removed, filtered_metadata_all

    def apply_lifespan_metadata(self, removed):
        for subdir in os.listdir(self.data_dir):
            if subdir not in removed: continue
            if subdir not in self.filtered_video_ids: continue

            meta_path = os.path.join(self.data_dir, subdir, "metadata.json")
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            to_rename = {item["entity"] for item in removed[subdir]}
            for entry in metadata:
                if "entity_tracking" not in entry:
                    continue
                for tracked in entry["entity_tracking"]:
                    if tracked.get("name") in to_rename:
                        tracked["name"] = "Unknown"

            save_path = meta_path.replace("metadata", "metadata_filtered")
            with open(save_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"[INFO] Updated metadata for {subdir}")

    def parse_majority_voting(self):
        with open(self.args.ann_path, 'r') as f:
            annotations = json.load(f)
        delete_result = {}
        keep_result = {}
        for filepath, model_votes in annotations.items():
            if self.args.video_type == "AVA":
                video_id = filepath.split("/")[-4]
            else:
                video_id = "/".join(filepath.split("/")[-5:-3])
            entity_id = filepath.split("/")[-2]
            delete_count = sum(v == "delete" for v in model_votes.values())
            if delete_count > len(model_votes) / 2:
                delete_result.setdefault(video_id, {}).setdefault(entity_id, []).append("/".join(filepath.split("/")[-2:]))
            else:
                keep_result.setdefault(video_id, {}).setdefault(entity_id, []).append("/".join(filepath.split("/")[-2:]))
        return delete_result, keep_result

    def apply_mv_entity_tracking(self, metadata, face_crop_dict):
        moved_count = 0
        metadata.setdefault("Unknown", [])
        for entity_id in list(metadata.keys()):
            if entity_id == "Unknown" or entity_id not in face_crop_dict:
                continue
            keep, drop = [], []
            crop_set = set(face_crop_dict[entity_id])
            for item in metadata[entity_id]:
                face_path = "/".join(item.get("face_save_path").split("/")[-2:])
                if face_path in crop_set:
                    drop.append(item)
                else:
                    keep.append(item)
            metadata[entity_id] = keep if keep else None
            metadata["Unknown"].extend(drop)
            moved_count += len(drop)
            if not keep:
                del metadata[entity_id]
        metadata["Unknown"].sort(key=lambda x: x["timestamp"])
        return metadata

    def apply_mv_metadata(self, metadata, face_crop_dict):
        for entry in metadata:
            if "entity_tracking" not in entry:
                continue
            for tracked in entry["entity_tracking"]:
                name = tracked.get("name")
                if name in face_crop_dict:
                    face_path = "/".join(tracked["face_save_path"].split("/")[-2:])
                    if face_path in face_crop_dict[name]:
                        tracked["name"] = "Unknown"
        return metadata

    def apply_majority_voting(self, delete_entities):
        for video_id, annotations in delete_entities.items():
            print(f"[INFO] Delete entities for {video_id}: {len(annotations)}")
            meta_path = os.path.join(self.data_dir, video_id, "metadata.json")
            et_meta_path = os.path.join(self.data_dir, video_id, "entity_tracking_metadata.json")
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            with open(et_meta_path, 'r') as f:
                et_metadata = json.load(f)

            updated_et = self.apply_mv_entity_tracking(et_metadata, annotations)
            updated_meta = self.apply_mv_metadata(metadata, annotations)

            with open(et_meta_path.replace("metadata", "metadata_filtered"), 'w') as f:
                json.dump(updated_et, f, indent=4)
            with open(meta_path.replace("metadata", "metadata_filtered"), 'w') as f:
                json.dump(updated_meta, f, indent=4)
            print(f"[INFO] Applied majority voting for {video_id}")

    def apply_keep_majority_voting(self, keep_entities):
        for video_id, annotations in keep_entities.items():
            print(f"[INFO] KEEP entities for {video_id}: {len(annotations)}")
            meta_path = os.path.join(self.data_dir, video_id, "metadata.json")
            et_meta_path = os.path.join(self.data_dir, video_id, "entity_tracking_metadata.json")
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            with open(et_meta_path, 'r') as f:
                et_metadata = json.load(f)
            with open(et_meta_path.replace("metadata", "metadata_filtered"), 'w') as f:
                json.dump(et_metadata, f, indent=4)
            with open(meta_path.replace("metadata", "metadata_filtered"), 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"[INFO] Applied majority voting for {video_id}")

    def parse_gemini_results(self, gemini_json_path, pre_annotation_folder="/mnt/task_wrapper/user_output/artifacts/run_pipeline/data_pipeline/owlbase_0.3/TVQA"):
        with open(gemini_json_path, "r") as f:
            gemini_data = json.load(f)

        gemini_annotations = defaultdict(dict)

        for video_id, entities in gemini_data.items():
            for entity_id, entries in entities.items():
                for entry in entries:
                    image_path = entry["image_path"]
                    # Relative to pre_annotation folder
                    rel_path = os.path.relpath(image_path, pre_annotation_folder)

                    try:
                        if isinstance(entry["results"], str):
                            results = json.loads(entry["results"])
                            same_identity = results.get("same_identity", None)
                        elif isinstance(entry["results"], dict):
                            same_identity = entry["results"].get("same_identity", None)

                        if same_identity is True:
                            label = "keep"
                        elif same_identity is False:
                            label = "delete"
                        else:
                            label = "unsure"
                    except Exception as e:
                        print(f"Warning: Failed to parse result for {image_path} due to {e}")
                        label = "unsure"

                    gemini_annotations[rel_path]["gemini"] = label
        return gemini_annotations

    def calc_iaa(self, result):
        annotators = list(next(iter(result.values())).keys())
        with open(self.args.ann_path.replace(".json", ".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image"] + annotators)
            for image, annots in result.items():
                row = [image] + [annots.get(a, "") for a in annotators]
                writer.writerow(row)

        df = pd.read_csv(self.args.ann_path.replace(".json", ".csv"), index_col=0)
        per_annotator_agreement = percentage_agreement(df)
        print(f"Percentage of unanimous annotations: {percentage_agreement(df):.2%}")

        # Fleiss’ Kappa
        fk = compute_fleiss_kappa(df)
        print(f"\nFleiss’ Kappa (all annotators): {fk:.3f}")
        
    def gather_annotation(self):
        result = defaultdict(dict)
        gemini_files = []
        for file in os.listdir(self.args.ann_path):
            if "final" in file:
                gemini_files.append(os.path.join(self.args.ann_path, file))

        all_gemini_annotations = defaultdict(dict)
        for path in gemini_files:
            annotator_name = Path(path).stem.split("_")[0]  # e.g., gemini-2.0-flash_final_annotation → use as annotator name
            if annotator_name == "annotation":
                annotator_name = Path(path).stem.split("annotation_")[-1].split("_")[0]
            annotations = self.parse_gemini_results(path)
            for rel_path, label_dict in annotations.items():
                all_gemini_annotations[rel_path][annotator_name] = label_dict["gemini"]      

        for rel_path, annot_dict in all_gemini_annotations.items():
            result[rel_path].update(annot_dict)

        self.args.ann_path = os.path.join(self.args.ann_path, "annotation_matrix.json")
        with open(self.args.ann_path, "w") as f:
            json.dump(result, f, indent=2)
        self.calc_iaa(result)

    def gather_detection_acc(self):
        matched_dirs = []
        target_filename = "detection_bbox_acc.json"
        avg_accs = []
        for dirpath, dirnames, filenames in os.walk(self.args.data_dir):
            if target_filename in filenames:
                full_path = os.path.join(dirpath, target_filename)
                with open(full_path, 'r') as f:
                    data = json.load(f)

                if self.args.video_type == "AVA":
                    pre = data['overall_precision']
                    recall = data['overall_recall']
                    detection_acc = (2 * pre * recall) / (pre + recall) if (pre + recall) > 0 else 0
                elif self.args.video_type in ["TVQA", "TRECKVID", "VideoMME", "LVBench"]:
                    if 'average_set_mIoU_valid' not in data: continue
                    detection_acc = data['average_set_mIoU_valid']

                if detection_acc >= self.args.detection_acc_thres:
                    matched_dirs.append(dirpath)
                avg_accs.append(detection_acc)
        print(f"[INFO] Average detection accuracy across all directories: {sum(avg_accs) / len(avg_accs) if avg_accs else 0:.2f}")
        print(f"[INFO] Found {len(matched_dirs)} directories with detection accuracy >= {self.args.detection_acc_thres}:")
        self.filtered_video_ids = [os.path.relpath(d, self.args.data_dir) for d in matched_dirs]
        print(self.filtered_video_ids)

    def run(self):
        if self.args.filter_type == "lifespan":
            removed, filtered_metadata_all = self.apply_lifespan_entity_tracking()
            self.apply_lifespan_metadata(removed)
        elif self.args.filter_type == "majority_voting":
            self.gather_annotation()
            delete_entities, keep_entities = self.parse_majority_voting()
            self.apply_majority_voting(delete_entities)
            self.apply_keep_majority_voting(keep_entities)
        elif self.args.filter_type == "detection_acc":
            self.gather_detection_acc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results/data_pipeline")
    parser.add_argument("--lifespan_thres", type=int, default=60)
    parser.add_argument("--consec_dur_thres", type=int, default=3)
    parser.add_argument("--video_type", type=str, default="AVA", choices=["AVA", "TVQA"])
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--filter_type", type=str, default="majority_voting", choices=["majority_voting", "lifespan"])
    parser.add_argument("--ann_path", type=str, default="results/annotation")
    # gemini annotation arguments
    parser.add_argument("--num_images_per_call", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash", choices=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])
    parser.add_argument("--refs_root", type=str, default="refs")
    parser.add_argument("--save_dir", type=str, default="results/annotations")
    parser.add_argument("--use_unsure", action="store_true")
    args = parser.parse_args()

    # lifespan filtering before gemini annotation
    args.filter_type = "lifespan"
    processor = PostProcess(args)
    processor.run()

    # gemini annotation
    annotator = GeminiAnnotation(args)
    annotator.run()

    # majority voting after gemini annotation
    args.filter_type = "majority_voting"
    processor = PostProcess(args)
    processor.run()
