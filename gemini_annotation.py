#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
from math import ceil
from collections import defaultdict

from pipeline.gemini import GeminiImageIdentityPipeline
from prompt import (
    GEMINI_ENTITY_TRACKING_PROMPT_WOUNSURE, GEMINI_ENTITY_TRACKING_PROMPT_UNSURE,
    GEMINI_ENTITY_TRACKING_PROMPT_WOUNSURE_MULTI, GEMINI_ENTITY_TRACKING_PROMPT_UNSURE_MULTI
)

class GeminiAnnotation:
    def __init__(self, args, filtered_video_ids=None):
        self.args = args
        self.face_ref_dir = os.path.join(args.face_ref_dir, args.video_type)
        self.data_dir = args.data_dir
        self.filtered_video_ids = filtered_video_ids
        self.save_dir = os.path.join(args.save_dir, "gemini_annotation", args.video_type, "UNSURE" if args.use_unsure else "WOUNSURE")
        os.makedirs(self.save_dir, exist_ok=True)
        print("Saving results to:", self.save_dir)

        self.engine = GeminiImageIdentityPipeline(args.gemini_model_name)
        self.num_images_per_call = args.num_images_per_call

        if self.num_images_per_call > 1:
            self.user_prompt = GEMINI_ENTITY_TRACKING_PROMPT_UNSURE_MULTI if args.use_unsure else GEMINI_ENTITY_TRACKING_PROMPT_WOUNSURE_MULTI
        else:
            self.user_prompt = GEMINI_ENTITY_TRACKING_PROMPT_UNSURE if args.use_unsure else GEMINI_ENTITY_TRACKING_PROMPT_WOUNSURE

    def extract_error_entries(self, gemini_json_path, output_path):
        with open(gemini_json_path, "r") as f:
            data = json.load(f)

        error_entries = {}
        for video_id, entities in data.items():
            error_entries[video_id] = defaultdict(list)
            for entity_id, entries in entities.items():
                for entry in entries:
                    target = entry.get("results", "").strip() if isinstance(entry.get("results"), str) else entry.get("results", "")
                    if target == "_ERROR_":
                        error_entries[video_id][entity_id].append(entry)

        with open(output_path, "w") as f:
            json.dump(error_entries, f, indent=2)

        return error_entries

    def build_inspection_json(self, output_path):
        data = {}
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if self.args.video_type == "AVA":
                    video_id = os.path.basename(root)
                else:
                    video_id = "/".join(root.split("/")[-2:])
                if video_id not in self.filtered_video_ids: continue
                print(f"Processing video: {video_id}")
                video_path = os.path.join(self.data_dir, video_id)
            
                data[video_id] = {}
                for entity_id in os.listdir(os.path.join(video_path, "face_recog")):
                    if "removed" in entity_id:
                        continue
                    entity_path = os.path.join(video_path, "face_recog", entity_id)
                    if not os.path.isdir(entity_path):
                        continue

                    print(f"  Processing entity: {entity_id}")
                    ref_img_path = os.path.join(self.face_ref_dir, video_id, f"{entity_id}.png")
                    if not os.path.exists(ref_img_path):
                        print(f"[WARNING] Reference image not found: {ref_img_path}")
                        continue

                    image_files = sorted([
                        f for f in os.listdir(entity_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])
                    if not image_files:
                        print(f"[WARNING] No face crops found for {entity_id}")
                        continue

                    num_chunks = ceil(len(image_files) / self.num_images_per_call)
                    comparisons = []

                    for i in range(num_chunks):
                        chunk_files = image_files[i * self.num_images_per_call : (i + 1) * self.num_images_per_call]
                        face_crop_paths = [os.path.join(entity_path, f) for f in chunk_files]
                        results = self.engine.generate_response(self.user_prompt, ref_img_path, face_crop_paths)
                        for img_file, result in zip(chunk_files, results):
                            comparisons.append({
                                "reference_image_path": ref_img_path,
                                "image_path": os.path.join(entity_path, img_file),
                                "results": result
                            })

                    if comparisons:
                        data[video_id][entity_id] = comparisons

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def build_inspection_error_entries_json(self, error_entries, output_path):
        data = {}
        for video_id, entities in error_entries.items():
            if video_id not in self.filtered_video_ids: continue

            data[video_id] = {}
            for entity_id, entries in entities.items():
                grouped_by_ref = defaultdict(list)
                for entry in entries:
                    ref_path = entry.get("reference_image_path")
                    face_path = entry.get("image_path")
                    if not ref_path or not face_path:
                        print(f"[WARNING] Missing paths in entry: {entry}")
                        continue
                    grouped_by_ref[ref_path].append(face_path)

                comparisons = []
                for ref_path, face_crop_paths in grouped_by_ref.items():
                    num_chunks = ceil(len(face_crop_paths) / self.num_images_per_call)
                    for i in range(num_chunks):
                        chunk = face_crop_paths[i * self.num_images_per_call : (i + 1) * self.num_images_per_call]
                        results = self.engine.generate_response(self.user_prompt, ref_path, chunk)
                        for crop_path, result in zip(chunk, results):
                            comparisons.append({
                                "reference_image_path": ref_path,
                                "face_crop_path": crop_path,
                                "results": result
                            })

                if comparisons:
                    data[video_id][entity_id] = comparisons

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def merge_regenerated_entries(self, original_path, retried_path, output_path):
        with open(original_path, "r") as f:
            original_data = json.load(f)
        with open(retried_path, "r") as f:
            retried_data = json.load(f)

        retried_lookup = {
            item["face_crop_path"]: item
            for video_id, entry in retried_data.items()
            for entity_id, items in entry.items()
            for item in items
        }

        updated_data = {}
        for video_id, entities in original_data.items():
            if video_id not in self.filtered_video_ids: continue
            
            updated_data[video_id] = {}
            for entity_id, entries in entities.items():
                updated_entries = []
                for entry in entries:
                    image_path = entry.get("image_path")
                    if entry.get("results", "") == "_ERROR_" and image_path in retried_lookup:
                        updated_entry = {
                            "reference_image_path": entry["reference_image_path"],
                            "image_path": image_path,
                            "results": retried_lookup[image_path]["results"]
                        }
                    else:
                        updated_entry = entry
                    updated_entries.append(updated_entry)
                updated_data[video_id][entity_id] = updated_entries

        with open(output_path, "w") as f:
            json.dump(updated_data, f, indent=2)

    def run(self):
        original_path = os.path.join(self.save_dir, f"annotation_{self.args.gemini_model_name}_nimgs{self.num_images_per_call}.json")
        self.build_inspection_json(original_path)

        error_path = os.path.join(self.save_dir, f"{self.args.gemini_model_name}_error_entries.json")
        error_entries = self.extract_error_entries(original_path, error_path)

        retried_path = os.path.join(self.save_dir, f"{self.args.gemini_model_name}_error_entries_annotation_nimgs{self.num_images_per_call}.json")
        self.build_inspection_error_entries_json(error_entries, retried_path)

        merged_output_path = os.path.join(self.save_dir, f"{self.args.gemini_model_name}_final_annotation_nimgs{self.num_images_per_call}.json")
        self.merge_regenerated_entries(original_path, retried_path, merged_output_path)

