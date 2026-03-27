#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import cv2
import glob
import random
import shutil
import json
import torch
import numpy as np
import argparse
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

from model_utils import load_fastreid_model, preprocess, EntityClustering, load_clip_model, compute_clip_embedding
from video_utils import get_all_videos, find_video_file_path
from info import VIDEO_ROOT
from utils import draw_distribution

from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg


class VideoFilter:
    def __init__(self, args):
        self.args = args
        self.video_type = args.video_type
        self.brightness_threshold = args.brightness_threshold
        self.save_dir = os.path.join(args.save_dir, "video_filtering")
        self.face_ref_dir = args.face_ref_dir
        self.video_dir = VIDEO_ROOT[self.video_type]
        os.makedirs(self.save_dir, exist_ok=True)

    def calculate_average_brightness(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.args.frame_rate = fps
        if not cap.isOpened():
            print(f"Failed to open: {video_path}")
            return None

        total_brightness = 0
        frame_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % fps == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                total_brightness += np.mean(gray)
                frame_count += 1
            frame_idx += 1

        cap.release()
        return total_brightness / frame_count if frame_count > 0 else None

    def _brightness_worker(self, video_path):
        avg_brightness = self.calculate_average_brightness(video_path)
        is_dark = avg_brightness is not None and avg_brightness < self.brightness_threshold
        return video_path, avg_brightness, is_dark

    def worker_main(self, assigned_videos, proc_idx):
        print(f"[Proc {proc_idx}] Processing {len(assigned_videos)} videos")
        with mp.Pool(processes=self.args.procs_per_gpu) as pool:
            results = list(tqdm(pool.imap(self._brightness_worker, assigned_videos), total=len(assigned_videos)))

        video_stats = {}
        for video_path, brightness, is_dark in results:
            if self.args.video_type in ["AVA", "OTHERS"]:
                video_id = os.path.basename(video_path).split(".")[0]
            else:
                video_id = "/".join(video_path.split("/")[-2:])
            video_stats[video_id] = {
                "brightness": brightness,
                "dark": "Yes" if is_dark else "No"
            }

        save_path = os.path.join(self.save_dir, f"{self.video_type}_proc{proc_idx}_stats.json")
        with open(save_path, "w") as f:
            json.dump(video_stats, f, indent=4)
        print(f"[Proc {proc_idx}] Saved {len(video_stats)} stats to {save_path}")

    @staticmethod
    def _launch_worker(args, video_list, proc_idx):
        vf = VideoFilter(args)
        vf.worker_main(video_list, proc_idx)

    def run_parallel_filtering(self):
        video_paths = get_all_videos(self.video_type, self.face_ref_dir, dark_video_filter=False)
        print(f"[INFO] Found {len(video_paths)} videos.")

        total_procs = self.args.num_gpus
        split_videos = [[] for _ in range(total_procs)]
        for i, vp in enumerate(video_paths):
            split_videos[i % total_procs].append(vp)

        processes = []
        for proc_idx in range(total_procs):
            p = mp.Process(target=self._launch_worker, args=(self.args, split_videos[proc_idx], proc_idx))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("[DONE] All CPU processes finished.")

    def merge_and_finalize_results(self):
        merged_stats = {}
        for fname in os.listdir(self.save_dir):
            if fname.startswith(self.video_type) and fname.endswith("_stats.json"):
                with open(os.path.join(self.save_dir, fname)) as f:
                    merged_stats.update(json.load(f))

        merged_path = os.path.join(self.save_dir, f"{self.video_type}_all_merged_stats.json")
        with open(merged_path, "w") as f:
            json.dump(merged_stats, f, indent=4)
        print(f"[INFO] Merged stats saved to: {merged_path}")

        non_dark_ids = [
            vid for vid, info in merged_stats.items()
            if info["dark"] == "No" and info["brightness"] is not None
        ]
        print(f"[INFO] Found {non_dark_ids} non-dark videos.")

        non_dark_paths, video_list = [], []
        for vid in non_dark_ids:
            if self.args.video_type in ["AVA"]:
                path = find_video_file_path(vid, self.video_dir)
                if path:
                    non_dark_paths.append(os.path.basename(path))
            else:
                path = os.path.join(self.video_dir, vid)
                non_dark_paths.append(vid)
            video_list.append(path)

        list_path = os.path.join(self.save_dir, f"{self.video_type}_non_dark_final_list.txt")
        with open(list_path, "w") as f:
            for name in sorted(non_dark_paths):
                f.write(name + "\n")

        print(f"[INFO] Final list saved to: {list_path}")
        print(f"[INFO] Total non-dark videos: {len(non_dark_paths)}")
        return video_list, merged_stats

    def get_stats(self, video_stats):
        values = [info["brightness"] for info in video_stats.values() if info["brightness"] is not None]
        draw_distribution(values, os.path.join(self.args.save_dir, f"{self.args.video_type}_brightness_distribution.png"))

class MainCharacterSelector:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = os.path.join(self.args.save_dir, "preprocess", self.args.method, self.args.video_type)
        self.face_ref_dir = os.path.join(self.args.face_ref_dir, self.args.video_type)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.face_ref_dir, exist_ok=True)
        self.frame_size_dict = {}
        self.clip_model, self.clip_preprocess = load_clip_model()

    def get_detector(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        return DefaultPredictor(cfg)

    def get_tracker(self):
        return EntityClustering(sim_threshold=self.args.sim_threshold)
    
    def detect(self, detector, frame):
        with torch.no_grad():
            results = detector(frame)
        mask = results["instances"].pred_classes.cpu() == 0
        boxes = results["instances"].pred_boxes.tensor[mask].cpu()
        confs = results["instances"].scores.cpu().numpy() 
        return boxes, confs

    def detect_and_track(self, detector, tracker, vid_writer, cap, save_dir, crop_save_root):
        frame_idx = 0
        metadata_by_frame = {}
        frame_idx = 0
        unique_ids = set()

        reid_model = load_fastreid_model()
        transform_reid = preprocess()

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % frame_rate != 0:
                frame_idx += 1
                continue
            if self.args.video_type == "OTHERS":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame_save = frame.copy()
            print(f"[INFO] Frame {frame_idx}")
            boxes, confs = self.detect(detector, frame)
            entities = []
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
        
                # Crop person image
                crop = frame[y1:y2, x1:x2]
                height, width = frame.shape[:2]
                if (x2-x1) < 0.5*width and (y2-y1) < 0.5*height: continue

                # ReID preprocessing
                crop_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                img_tensor = transform_reid(crop_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = reid_model(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                emb = features.cpu().numpy()[0]

                # Add to clustering
                person_id = f"{frame_idx:06d}_{x1}_{y1}"
                entity_id, is_new = tracker.add_person(person_id=f"{frame_idx:06d}_{x1:.0f}_{y1:.0f}", person_embedding=emb)

                crop_save_dir = os.path.join(crop_save_root, str(entity_id))
                os.makedirs(crop_save_dir, exist_ok=True)
                crop_img.save(os.path.join(crop_save_dir, f"{person_id}.png"))

                # Save entity info
                entities.append({
                    "id": int(entity_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(conf)
                })
                unique_ids.add(int(entity_id))

                # Draw
                label = f"{entity_id}"
                cv2.rectangle(frame_save, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_save, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            vid_writer.write(frame_save)
            metadata_by_frame[f"{frame_idx:06d}"] = entities
            frame_idx += 1

        cap.release()
        vid_writer.release()

        cluster_path = os.path.join(save_dir, "entity_clusters.json")
        entity_clusters_serializable = [sorted(list(cluster)) for cluster in tracker.entity_clusters]
        with open(cluster_path, "w") as f:
            json.dump(entity_clusters_serializable, f, indent=2)

        return metadata_by_frame, unique_ids

    def run_detect_and_track(self, video_path, detector, tracker):
        print("[INFO] Processing video:", video_path)
        cap = cv2.VideoCapture(video_path)
        if self.args.video_type in ["AVA", "OTHERS"]:
            video_id = os.path.basename(video_path).split('.')[0]
        elif self.args.video_type in ["TVQA", "TRECVID", "VideoMME", "LVBench"]:
            video_id = "/".join(video_path.split("/")[-2:])
            video_id = video_id.split(".")[0]
        
        sub_dir = f"{self.args.detector_type}_{self.args.tracker_type}"
        save_dir = os.path.join(self.save_dir, sub_dir, video_id)
        os.makedirs(save_dir, exist_ok=True)
        save_video_path = os.path.join(save_dir, self.args.save_video_name)
        save_path = os.path.join(save_dir, self.args.save_metadata_name)
        crop_save_root = os.path.join(save_dir, "person")
        os.makedirs(crop_save_root, exist_ok=True)
        print(f"[INFO] Saving results to {save_path}")
        print(f"[INFO] Saving video to {save_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        self.frame_size_dict[video_id] = (W, H)

        metadata_by_frame, unique_ids = self.detect_and_track(detector, tracker, vid_writer, cap, save_dir, crop_save_root)
        
        with open(save_path, "w") as f:
            json.dump(metadata_by_frame, f, indent=2)

        print(f"[DONE] ReID-based entity metadata saved to {save_path}")
        print(f"[DONE] Unique IDs: {len(unique_ids)}")
        return video_id, save_dir, crop_save_root, fps
    
    def compute_consecutive_durations(self, timestamps):
        """
        Given a sorted list of timestamps, return the list of consecutive durations.
        """
        if not timestamps:
            return []

        timestamps = sorted(timestamps)
        durations = []
        start = timestamps[0]
        prev = timestamps[0]

        for t in timestamps[1:]:
            if t == prev + 1:
                prev = t
            else:
                durations.append(prev - start + 1)
                start = t
                prev = t
        durations.append(prev - start + 1)
        return durations

    def save_top_4_groups(self, crop_dir, output_dir, fps):
        os.makedirs(output_dir, exist_ok=True)
        group_counts = {}
        removed = {}

        for entity in os.listdir(crop_dir):
            entity_path = os.path.join(crop_dir, entity)
            if not os.path.isdir(entity_path):
                continue
            
            frame_indices = []
            for f in os.listdir(entity_path):
                if f.endswith('.png'):
                    frame_idx = int(f.split('_')[0])
                    frame_indices.append(frame_idx)
            if not frame_indices: continue
            timestamps = [idx/fps for idx in frame_indices]
            lifespan = max(timestamps) - min(timestamps) + 1
            consecutive_durations = self.compute_consecutive_durations(timestamps)
            has_valid_segment = any(d >= self.args.consec_dur_thres for d in consecutive_durations)

            if lifespan >= self.args.lifespan_thres and has_valid_segment:
                image_count = len(frame_indices)
                group_counts[entity] = image_count
            else:
                removed.setdefault(entity, []).append({
                    "lifespan": lifespan,
                    "has_valid_segment": has_valid_segment
                })
                print(f"[WARNING] Removed {entity} with lifespan {lifespan} and valid segment {has_valid_segment}")

        # Sort by number of images
        sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"[INFO] Top-4 groups:")
        for rank, (name, count) in enumerate(sorted_groups):
            print(f"  {name}: {count} images")
            src = os.path.join(crop_dir, name)
            dst = os.path.join(output_dir, f"rank_{rank}")
            os.makedirs(dst, exist_ok=True)

            # Copy top images (you can also sort by filename if needed)
            for img_file in os.listdir(src):
                if img_file.endswith(".png"):
                    src_path = os.path.join(src, img_file)
                    dst_path = os.path.join(dst, img_file)
                    os.system(f"cp '{src_path}' '{dst_path}'")

        with open(os.path.join(output_dir, "removed_entities.json"), "w") as f:
            json.dump(removed, f, indent=2)
        print(f"[DONE] Saved top-4 entity crops to: {output_dir}")
    
    def filter_similar_entity(self, ref_dir):
        entity_paths = sorted(glob.glob(os.path.join(ref_dir, "entity_*.png")))
        if len(entity_paths) < 2: return  # Nothing to filter

        embeddings = {}
        for path in entity_paths:
            embeddings[path] = compute_clip_embedding(self.clip_model, self.clip_preprocess, path)

        # Step 2: Compute pairwise similarity
        to_remove = set()
        pairs = list(combinations(entity_paths, 2))
        for p1, p2 in pairs:
            sim = cosine_similarity([embeddings[p1]], [embeddings[p2]])[0][0]
            print(f"[INFO] Similarity between {os.path.basename(p1)} and {os.path.basename(p2)}: {sim:.2f}")
            if sim > 0.9:  
                print(f"[INFO] Removing one of: {p1} vs {p2} (sim={sim:.2f})")
                to_remove.add(p2)  # arbitrarily keep p1

        # Step 3: Remove redundant images
        for path in to_remove:
            os.remove(path)

    def select_rep_image(self, video_id, top_output_dir):
        ref_save_dir = os.path.join(self.face_ref_dir, video_id)
        os.makedirs(ref_save_dir, exist_ok=True)
        orig_w, orig_h = self.frame_size_dict[video_id]

        for rank in range(4):
            group_dir = os.path.join(top_output_dir, f"rank_{rank}")
            if not os.path.exists(group_dir):
                print(f"[Warning] Missing group directory: {group_dir}")
                continue

            image_paths = sorted(glob.glob(os.path.join(group_dir, "*.jpg")) + glob.glob(os.path.join(group_dir, "*.png")))
            if not image_paths:
                print(f"[Warning] No images found in {group_dir}")
                continue

            # Filter images by size constraint
            valid_images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is None: continue
                    
                h, w = img.shape[:2]
                if h > 0.5 * orig_h and w > 0.5 * orig_w:
                    valid_images.append(path)

            if not valid_images:
                print(f"[Warning] No valid images (size constraint) in group {rank}")
                continue

            rep_img_path = random.choice(valid_images)
            rep_out_path = os.path.join(ref_save_dir, f"entity_{rank}.png")
            shutil.copy(rep_img_path, rep_out_path)
        return ref_save_dir

    def run_single_video(self, video_path, device_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            detector = self.get_detector()
            tracker = self.get_tracker()
            video_id, save_dir, crop_save_root, fps = self.run_detect_and_track(video_path, detector, tracker)
            top_output_dir = os.path.join(save_dir, "top_face_groups")
            self.save_top_4_groups(crop_save_root, top_output_dir, fps)
            ref_save_dir = self.select_rep_image(video_id, top_output_dir)
            self.filter_similar_entity(ref_save_dir)
        except Exception as e:
            print(f"[GPU {device_id}] Failed on {video_path}: {e}")

    def worker(self, proc_idx, assigned_videos, num_gpus):
        gpu_id = proc_idx % num_gpus
        for video_path in tqdm(assigned_videos, desc=f"[GPU {gpu_id} | Proc {proc_idx}]"):
            self.run_single_video(video_path, gpu_id)

    def run_parallel_tracking(self, video_paths, num_gpus, procs_per_gpu):
        total_procs = num_gpus * procs_per_gpu
        split_video_lists = [[] for _ in range(total_procs)]
        for idx, vid in enumerate(video_paths):
            split_video_lists[idx % total_procs].append(vid)

        processes = []
        for proc_idx in range(total_procs):
            p = mp.Process(target=self.worker, args=(proc_idx, split_video_lists[proc_idx], num_gpus))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def run(self, video_paths):
        print(f"[INFO] Found {len(video_paths)} videos")
        self.run_parallel_tracking(video_paths, self.args.num_gpus, self.args.procs_per_gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dark videos in a directory.")
    parser.add_argument("--video_type", type=str, default="AVA", choices=["AVA", "TVQA"])
    parser.add_argument("--brightness_threshold", type=float, default=50.0)
    parser.add_argument("--face_ref_dir", type=str, default="refs")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--detector_type", type=str, default="detectron2")
    parser.add_argument("--tracker_type", type=str, default="reid")
    parser.add_argument("--method", type=str, default="detect_track")
    parser.add_argument("--lifespan_thres", type=int, default=60, help="Minimum lifespan of entities in seconds")
    parser.add_argument("--consec_dur_thres", type=int, default=3, help="Minimum consecutive duration in seconds")
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--sim_threshold", type=float, default=0.75)
    parser.add_argument("--save_video_name", type=str, default="output_video.mp4")
    parser.add_argument("--save_metadata_name", type=str, default="metadata.json")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--procs_per_gpu", type=int, default=1)

    args = parser.parse_args()
    if args.video_type == "AVA": 
        args.frame_rate = 30
    elif args.video_type == "TVQA":
        args.frame_rate = 3

    from video_utils import get_all_videos
    video_list = get_all_videos(args.video_type, args.face_ref_dir, dark_video_filter=True, filtered_video_path=os.path.join(args.save_dir, 'video_filtering'))

    # Select main characters on non-dark videos
    print("[INFO] Starting main character selection...")
    save_dir = os.path.join(args.save_dir, "preprocess", args.method, args.video_type)
    os.makedirs(save_dir, exist_ok=True)
    # mp.set_start_method("spawn", force=True)
    selector = MainCharacterSelector(args)
    for video_path in video_list:
        selector.run_single_video(video_path, device_id=0)