#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import cv2
import torch
import multiprocessing as mp
from pathlib import Path
import time
import json
from PIL import Image
import numpy as np
import face_recognition
from tqdm import tqdm
from collections import defaultdict
from torchvision import ops

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from info import FRAME_ROOT
from model_utils import (
    get_detector, build_face_db, detect_person,
    detect_person_owlv2, get_owlv2_detector
)
from utils import convert_numpy, crop_outfit_region, save_outfit_crop
from video_utils import (
    load_ava_annotations, extract_bboxes_for_timestamp,
)
from metrics import (
    compute_set_miou, calc_consecutive_duration,
    evaluate_entity_detection_per_timestamp, evaluate_detection_per_timestamp,
)

class EntityTracking:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.use_owl:
            print(f"[INFO] Using OWLv2 model type: {args.owl_model_type} with threshold: {args.owl_thres}")
            self.save_dir = os.path.join(f"{args.save_dir}", "data_pipeline", f"owl{args.owl_model_type}_{args.owl_thres}", args.video_type)
        else:
            self.save_dir = os.path.join(f"{args.save_dir}", "data_pipeline", "detectron2", args.video_type)
        os.makedirs(self.save_dir, exist_ok=True)

    def setup(self, video_id):
        save_dir = os.path.join(self.save_dir, video_id)
        detection_dir = os.path.join(save_dir, "detections")
        face_dir = os.path.join(save_dir, "face_recog")
        outfit_dir = os.path.join(save_dir, "outfits")
        overlay_dir = os.path.join(save_dir, "overlays")
        for d in [face_dir, outfit_dir, detection_dir, overlay_dir]:
            os.makedirs(d, exist_ok=True)
        return save_dir, detection_dir, face_dir, outfit_dir, overlay_dir

    def merge_boxes(self, person_boxes, owl_person_boxes, iou_threshold=0.5):
        """
        person_boxes: Tensor[N, 4] from Detectron2 (x1, y1, x2, y2)
        owl_person_boxes: Tensor[M, 4] from OWLv2 (x1, y1, x2, y2)
        iou_threshold: float — boxes with IoU > threshold are removed from OWLv2 set

        Returns:
            unified_boxes: Tensor[P, 4] — union of both sets, excluding high-overlap OWLv2 boxes
        """
        if person_boxes.size(0) == 0:
            return owl_person_boxes, 1  # nothing to compare with — keep all OWLv2 boxes
        if owl_person_boxes.size(0) == 0:
            print("[INFO] No OWLv2 detections found, using only Detectron2 boxes")
            return person_boxes, 0  # no OWLv2 detections

        # Compute IoU matrix [M, N]
        ious = ops.box_iou(owl_person_boxes, person_boxes)  # (M, N)

        # Get max IoU for each OWLv2 box
        max_ious = ious.max(dim=1).values

        # Mask: keep only OWLv2 boxes with IoU <= threshold
        keep_mask = max_ious <= iou_threshold
        filtered_owl_boxes = owl_person_boxes[keep_mask]

        # Combine the boxes
        unified_boxes = torch.cat([person_boxes, filtered_owl_boxes], dim=0)
        if len(unified_boxes) > len(person_boxes):
            print("[INFO] additional boxes extracted from OWLv2")
        return unified_boxes, int(len(unified_boxes) > len(person_boxes))

    def entity_detection(self, ava_df, video_id, timestamp, img_size, frame_save, frame, detector, owl_processor=None, owl_detector=None):
        gt_boxes = []
        if ava_df is not None:
            gt_boxes = extract_bboxes_for_timestamp(ava_df, video_id, timestamp, img_size)
            for gt in gt_boxes:
                x1, y1, x2, y2 = gt["bbox"]
                cv2.rectangle(frame_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_save, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        detectron2_person_boxes, _ = detect_person(detector, frame)
        if self.args.use_owl:
            assert owl_processor is not None and owl_detector is not None, "OWLv2 processor and detector must be provided"
            owl_person_boxes, _ = detect_person_owlv2(owl_processor, owl_detector, frame, threshold=self.args.owl_thres)
            set_miou = compute_set_miou(detectron2_person_boxes, owl_person_boxes)

            person_boxes, add_from_owl = self.merge_boxes(detectron2_person_boxes, owl_person_boxes)
            return person_boxes, gt_boxes, add_from_owl, set_miou
        else:
            return detectron2_person_boxes, gt_boxes, 0, 0  # No OWLv2 boxes to merge

    def entity_tracking(self, timestamp, frame, frame_save, person_boxes, gt_boxes, face_db, detection_dir, face_dir, outfit_dir, entity_tracking_metadata):
        entity_trackings = []
        names = []
        for box_id, box in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, box)
            # Sanitize crop region
            H, W = frame.shape[:2]
            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))
            
            cv2.rectangle(frame_save, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_save, "Pred", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            person_crop = frame[y1:y2, x1:x2]
            crop_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            detection_save_path = os.path.join(detection_dir, f"{timestamp:06d}_box{box_id:03d}.png")
            crop_img.save(detection_save_path)

            rgb_frame = person_crop[:, :, ::-1].astype(np.uint8)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_box = "not_detected"
            face_save_path = "not_detected"
            outfit_save_path = "not_detected"
            outfit_bbox = "not_detected"
            name = "Unknown"

            if face_encodings:
                distances = face_recognition.face_distance(face_db["encodings"], face_encodings[0])
                matches = face_recognition.compare_faces(face_db["encodings"], face_encodings[0], tolerance=0.5)
                if any(matches):
                    best_idx = np.argmin(distances)
                    name = face_db["names"][best_idx]
                    f_top, f_right, f_bottom, f_left = face_locations[0]
                    f_top += y1
                    f_bottom += y1
                    f_left += x1
                    f_right += x1
                    face_box = [f_left, f_top, f_right, f_bottom]
                    
                    # Save face
                    face_crop = frame[f_top:f_bottom, f_left:f_right]
                    face_save_path = os.path.join(face_dir, name, f"{timestamp:06d}_box{box_id:03d}.jpg")
                    os.makedirs(os.path.dirname(face_save_path), exist_ok=True)
                    cv2.imwrite(face_save_path, face_crop)

                    # Save outfit
                    outfit_crop, outfit_bbox = crop_outfit_region(frame, [x1, y1, x2, y2], face_box)
                    outfit_save_path = save_outfit_crop(outfit_dir, name, timestamp, box_id, outfit_crop)

            # Store tracking result
            entity_tracking_result = {
                "person_bbox": [x1, y1, x2, y2],
                "outfit_bbox": outfit_bbox,
                "face_bbox": face_box,
                "detection_save_path": detection_save_path,
                "face_save_path": face_save_path,
                "outfit_save_path": outfit_save_path,
            }

            entity_trackings.append({"name": name, **entity_tracking_result})
            entity_tracking_metadata[name].append({"timestamp": timestamp, **entity_tracking_result, "gt": gt_boxes})
            if name != "Unknown": names.append(name)
        return entity_trackings, entity_tracking_metadata, names

    def run_pipeline(self, video_path, device):
        print(f"[INFO] Processing video: {video_path} | face_ref_dir: {self.args.face_ref_dir}")
        args = self.args
        start_time = time.time()
        set_mious = []

        if args.video_type == "AVA":
            video_id = Path(video_path).stem
            ava_df = load_ava_annotations(args.ava_csv_path)
            start_sec = 900
        elif args.video_type == "OTHERS":
            video_id = Path(video_path).stem
            video_id = video_id.replace(".mov", "")
            ava_df = None
            start_sec = 0
        else:
            video_id = "/".join(video_path.split("/")[-2:])
            video_id = video_id.replace(".mp4", "")
            ava_df = None
            start_sec = 0
        print("load ava done")
        save_dir, detection_dir, face_dir, outfit_dir, overlay_dir = self.setup(video_id)
        detector = get_detector(args, device)
        if args.use_owl:
            owl_processor, owl_detector = get_owlv2_detector(args, device)
        else:
            owl_processor, owl_detector = None, None
        
        flip = True if self.args.video_type == "OTHERS" else False 
        face_db = build_face_db(os.path.join(args.face_ref_dir, args.video_type), video_id, face_dir)

        entity_tracking_metadata = defaultdict(list)
        detections, metadata = [], []
        cnt_owl = 0 
        owl_detections = defaultdict(list)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if flip: frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame_idx % fps != 0: 
                frame_idx += 1
                continue
            timestamp = frame_idx // fps + start_sec
            if args.video_type == "AVA" and timestamp < 902: 
                frame_idx += 1
                continue
                
            frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_size = frame_img.size
            if args.video_type == "AVA":
                frame_filename = f"{video_id}_{frame_idx+1:06d}.jpg"
            else:
                frame_filename = f"{frame_idx+1:05d}.jpg"
            print(f"[INFO] Time: {timestamp}, frame path: {frame_filename}")
            frame_path = os.path.join(FRAME_ROOT[args.video_type], video_id, frame_filename)
            frame_save = frame.copy()

            # Scene classification
            # scene_preds = scene_model.classify(frame)
            
            # Entity detection
            person_boxes, gt_boxes, add_from_owl, set_miou = self.entity_detection(
                ava_df, video_id, timestamp, img_size, frame_save, frame, detector, owl_processor, owl_detector
            )
            cnt_owl += add_from_owl
            set_mious.append(set_miou)
            detections.append({"timestamp": timestamp, "gt_boxes": gt_boxes, "pred_boxes": person_boxes})
            curr_metadata = {   
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "frame_path": frame_path,
                # "scene": scene_preds,
                "entity_detection": {
                    "gt_boxes": gt_boxes,
                    "pred_boxes": person_boxes,
                    "set_miou": set_miou
                }
            }
            
            # entity tracking
            entity_trackings, entity_tracking_metadata, names = self.entity_tracking(
                timestamp, frame, frame_save, person_boxes, gt_boxes, face_db, 
                detection_dir, face_dir, outfit_dir, entity_tracking_metadata
            )
                    
            metadata.append({**curr_metadata, "entity_tracking": entity_trackings})
            overlay_save_path = os.path.join(overlay_dir, f"{timestamp:06d}.jpg")
            cv2.imwrite(overlay_save_path, frame_save)
            if self.args.use_owl and add_from_owl:
                for name in names:
                    owl_detections[name].append(overlay_save_path)
            frame_idx += 1
        cap.release()

        if self.args.use_owl:
            print(f"[INFO] {cnt_owl} additional boxes from OWLv2")
            json.dump(owl_detections, open(os.path.join(save_dir, "owl_detection.json"), "w"), indent=4)
        
        avg_set_miou = np.mean(set_mious) if set_mious else 0
        results_detection = evaluate_detection_per_timestamp(detections, avg_set_miou)
        results_entity_detection = evaluate_entity_detection_per_timestamp(entity_tracking_metadata)
        duration_stats = calc_consecutive_duration(entity_tracking_metadata)

        json.dump(convert_numpy(metadata), open(os.path.join(save_dir, "metadata.json"), "w"), indent=4)
        json.dump(convert_numpy(detections), open(os.path.join(save_dir, "detection_bbox.json"), "w"), indent=4)
        json.dump(convert_numpy(results_detection), open(os.path.join(save_dir, "detection_bbox_acc.json"), "w"), indent=4)
        json.dump(convert_numpy(entity_tracking_metadata), open(os.path.join(save_dir, "entity_tracking_metadata.json"), "w"), indent=4)
        json.dump(convert_numpy(results_entity_detection), open(os.path.join(save_dir, "entity_detection_bbox_acc.json"), "w"), indent=4)
        json.dump(convert_numpy(duration_stats), open(os.path.join(save_dir, "duration_stats.json"), "w"), indent=4)
        print(f"[INFO] Processed {video_id} in {time.time() - start_time:.2f} seconds")

    def process_video_on_gpu(self, video_path, device_id):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        video_id = Path(video_path).stem
        print(f"[GPU {device_id}] Processing {video_id} on {device}...")
        try:
            self.run_pipeline(video_path, device)
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {video_path}: {e}")

    def worker(self, proc_idx, assigned_videos, num_gpus):
        gpu_id = proc_idx % num_gpus
        for video_path in tqdm(assigned_videos, desc=f"[GPU {gpu_id} | Proc {proc_idx}]"):
            self.process_video_on_gpu(video_path, gpu_id)

    def run_parallel_processing(self, video_paths):
        total_procs = self.args.num_gpus * self.args.procs_per_gpu
        split_video_lists = [[] for _ in range(total_procs)]
        for idx, video in enumerate(video_paths):
            split_video_lists[idx % total_procs].append(video)

        processes = []
        for proc_idx in range(total_procs):
            p = mp.Process(target=self.worker, args=(proc_idx, split_video_lists[proc_idx], self.args.num_gpus))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def run(self, video_paths):
        print(f"[INFO] Found {len(video_paths)} videos. Starting with {self.args.num_gpus} GPUs...")
        self.run_parallel_processing(video_paths)
