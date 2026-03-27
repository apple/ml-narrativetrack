#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import cv2
import json
from collections import defaultdict
import multiprocessing as mp

from video_utils import find_video_file_path
from info import VIDEO_ROOT
from chunk_utils import (
    dynamic_chunk_reappearing,
    dynamic_chunk_disappearing,
    dynamic_chunk_appearing,
    build_entity_trajectories
)

class VideoExtractor:
    def __init__(self, args):
        self.args = args
        self.args.frame_rate = 30 if args.video_type == "AVA" else 3
        self.metadata_dir = args.metadata_dir
        self.MIN_DURATION_FRAMES = args.min_duration * self.args.frame_rate
        self.save_dir = os.path.join(self.args.save_dir, f"extract_video_{self.args.min_duration}", self.args.video_type)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_all_metadata_paths(self):
        metadata_paths = []
        for dirpath, _, filenames in os.walk(self.metadata_dir):
            for fname in filenames:
                if fname.endswith(".json") and fname.startswith("metadata_filtered"):
                    metadata_paths.append(os.path.join(dirpath, fname))
        return metadata_paths

    @staticmethod
    def process_metadata_path(args_tuple):
        metadata_path, args, save_dir = args_tuple
        print(f"Processing metadata: {metadata_path}")
        if args.video_type == "AVA":
            video_id = os.path.basename(os.path.dirname(metadata_path))
        elif args.video_type in ["VideoMME", "LVBench"]:
            video_id = metadata_path.split("/")[-2]
        else:
            video_id = "/".join(metadata_path.split("/")[-3:-1])

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        trajectories, num_entities = build_entity_trajectories(metadata)
        
        video_path = find_video_file_path(video_id, base_dir=VIDEO_ROOT.get(args.video_type, ""))
        assert os.path.exists(video_path), f"Video path does not exist: {video_path}"
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video FPS: {fps}, Expected FPS: {video_id}")

        reappear_segments, num_reappear, avg_reappear = dynamic_chunk_reappearing(
            args, save_dir, video_id, metadata, trajectories,
            min_duration=args.min_duration, fps=fps)

        disappear_segments, num_disappear, avg_disappear = dynamic_chunk_disappearing(
            args, save_dir, video_id, metadata, trajectories,
            min_duration=args.min_duration, fps=fps)

        appear_segments, num_appear, avg_appear = dynamic_chunk_appearing(
            args, save_dir, video_id, metadata, trajectories,
            min_duration=args.min_duration, fps=fps)

        return {
            "video_id": video_id,
            "reappear": (reappear_segments, num_reappear, avg_reappear),
            "disappear": (disappear_segments, num_disappear, avg_disappear),
            "appear": (appear_segments, num_appear, avg_appear),
            "num_entities": num_entities,
        }

    def run(self):
        metadata_paths = self.get_all_metadata_paths()
        print(f"Found {len(metadata_paths)} metadata files in {self.metadata_dir}")

        args_list = [(p, self.args, self.save_dir) for p in metadata_paths]

        with mp.Pool(processes=self.args.procs_per_gpu) as pool:
            results = pool.map(VideoExtractor.process_metadata_path, args_list)

        total_reappear = total_disappear = total_appear = total_entity = 0
        avg_reappear_total = avg_disappear_total = avg_appear_total = avg_all = 0
        all_video_chunks = defaultdict(dict)

        for res in results:
            video_id = res["video_id"]
            reappear, disappear, appear = res["reappear"], res["disappear"], res["appear"]

            all_video_chunks[video_id]["reappear"] = reappear[0]
            all_video_chunks[video_id]["disappear"] = disappear[0]
            all_video_chunks[video_id]["appear"] = appear[0]

            total_reappear += reappear[1]
            total_disappear += disappear[1]
            total_appear += appear[1]
            total_entity += res["num_entities"]

            avg_reappear_total += reappear[1] * reappear[2]
            avg_disappear_total += disappear[1] * disappear[2]
            avg_appear_total += appear[1] * appear[2]
            avg_all += (
                reappear[1] * reappear[2] +
                disappear[1] * disappear[2] +
                appear[1] * appear[2]
            )

        print(f"Reappearing segments: {total_reappear}")
        print(f"Disappearing segments: {total_disappear}")
        print(f"Appearing segments: {total_appear}")
        print(f"Total segments processed: {total_reappear + total_disappear + total_appear}")
        print(f"Total entities processed: {total_entity}")
        print(f"Average reappearing segment duration: {avg_reappear_total / total_reappear if total_reappear else 0:.2f} seconds")
        print(f"Average disappearing segment duration: {avg_disappear_total / total_disappear if total_disappear else 0:.2f} seconds")
        print(f"Average appearing segment duration: {avg_appear_total / total_appear if total_appear else 0:.2f} seconds")
        print(f"Average duration of all segments: {avg_all / (total_reappear + total_disappear + total_appear) if (total_reappear + total_disappear + total_appear) else 0:.2f} seconds")

        with open(os.path.join(self.save_dir, "video_chunks_metadata.json"), "w") as f:
            json.dump(all_video_chunks, f, indent=4)

