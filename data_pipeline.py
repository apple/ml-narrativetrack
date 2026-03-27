#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import torch
import argparse
import multiprocessing as mp

from preprocess import VideoFilter, MainCharacterSelector
from entity_tracking import EntityTracking
from postprocess import PostProcess
from gemini_annotation import GeminiAnnotation
from extract_video import VideoExtractor
from recognition import GeminiActionRecognizer
from prompt import GEMINI_ENTIRE_RECOG_PROMPT, GEMINI_SINGLE_ACTION_RECOG_PROMPT

class DataPipeline:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Run full data pipeline.")
        self._add_arguments()
        self.args = self.parser.parse_args()

    def _add_arguments(self):
        # Shared
        self.parser.add_argument("--stage", nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7])
        self.parser.add_argument("--video_type", type=str, default="AVA", choices=["AVA", "TVQA", "OTHERS", "TRECVID", "VideoMME", "LVBench"])
        self.parser.add_argument("--frame_rate", type=int, default=30)
        self.parser.add_argument("--save_dir", type=str, default="results")
        self.parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
        self.parser.add_argument("--procs_per_gpu", type=int, default=1)

        # Video filter
        self.parser.add_argument("--brightness_threshold", type=float, default=50.0)
        self.parser.add_argument("--face_ref_dir", type=str, default="refs")

        # Character selection
        self.parser.add_argument("--detector_type", type=str, default="detectron2")
        self.parser.add_argument("--tracker_type", type=str, default="reid")
        self.parser.add_argument("--method", type=str, default="detect_track")
        self.parser.add_argument("--sim_threshold", type=float, default=0.75)
        self.parser.add_argument("--save_video_name", type=str, default="output_video.mp4")
        self.parser.add_argument("--save_metadata_name", type=str, default="metadata.json")

        # Post processing
        self.parser.add_argument("--detection_acc_thres", type=float, default=0.6)
        self.parser.add_argument("--lifespan_thres", type=int, default=60)
        self.parser.add_argument("--consec_dur_thres", type=int, default=3)
        
        # Entity tracking
        self.parser.add_argument("--use_owl", action="store_true")
        self.parser.add_argument("--owl_model_type", type=str, default="base", choices=["base", "large"])
        self.parser.add_argument("--owl_thres", type=float, default=0.3)
        self.parser.add_argument("--ava_csv_path", type=str, default="")
        
        # Annotation
        self.parser.add_argument("--data_dir", type=str, default="results/data_pipeline")
        self.parser.add_argument("--filter_type", type=str, default="majority_voting")
        self.parser.add_argument("--ann_path", type=str, default="results/annotation/annotation_matrix.json")
        self.parser.add_argument("--num_images_per_call", type=int, default=5)
        self.parser.add_argument("--gemini_model_name", type=str, default="gemini-2.0-flash", choices=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])
        self.parser.add_argument("--use_unsure", action="store_true")

        # Segment extraction
        self.parser.add_argument("--metadata_dir", type=str, default="results/data_pipeline")
        self.parser.add_argument("--min_duration", type=int, default=30)
        self.parser.add_argument("--reappear_gap_thres", type=int, default=2)

        # Action recog / appearance
        self.parser.add_argument("--video_root", type=str, default="narrahalluc_videos")
        self.parser.add_argument("--result_path", type=str, default="gemini_action_recognition_result.json")

    def run(self):
        mp.set_start_method("spawn", force=True)

        vf = VideoFilter(self.args)
        if 1 in self.args.stage:
            print("[1/6] Filtering dark videos...")
            self.args.procs_per_gpu = 4
            vf.run_parallel_filtering()
            video_list, merged_stats = vf.merge_and_finalize_results()
            vf.get_stats(merged_stats)
            print(f"Video frame rate: ", vf.args.frame_rate)
            self.args.frame_rate = vf.args.frame_rate
        else:
            print("[1/6] Loading the predefined non-dark videos....")
            from video_utils import get_all_videos
            video_list = get_all_videos(self.args.video_type, self.args.face_ref_dir, stage=self.args.stage[0], dark_video_filter=True, filtered_video_path=os.path.join(self.args.save_dir, 'video_filtering'))
        print(f"Filtered video list: {len(video_list)} videos.")
        
        selector = MainCharacterSelector(self.args)
        if 2 in self.args.stage:
            print(f"[2/6] Selecting main character for {len(video_list)}.")
            self.args.procs_per_gpu = 1
            selector.run(video_list)
        
        tracker = EntityTracking(self.args)
        if 3 in self.args.stage:
            print("[3/6] Running entity tracking...")
            tracker.run(video_list) # TODO: save randomly selected image per entity into 'refs'/ directory

        if 4 in self.args.stage:
            print("[4/6] Running annotation pipeline...")
            # Pre-annotation filter
            self.args.filter_type = "detection_acc"
            self.args.data_dir = tracker.save_dir
            detection_acc_filter = PostProcess(self.args)
            detection_acc_filter.run()
            filtered_video_ids = detection_acc_filter.filtered_video_ids
            print(f"Filtered video IDs: {len(filtered_video_ids)}")

            self.args.filter_type = "lifespan"
            PostProcess(self.args, filtered_video_ids=filtered_video_ids).run()

            gemini_annotators = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
            for gemini_model in gemini_annotators:
                print(f"[4/6] Running Gemini annotation with model: {gemini_model}...")
                self.args.gemini_model_name = gemini_model
                annotator = GeminiAnnotation(self.args, filtered_video_ids=filtered_video_ids)
                annotator.run()

            # Post-annotation voting
            self.args.filter_type = "majority_voting"
            self.args.ann_path = annotator.save_dir
            PostProcess(self.args).run()

        self.args.metadata_dir = tracker.save_dir
        extractor = VideoExtractor(self.args)
        if 5 in self.args.stage:
            print("[5/6] Extracting video segments...")
            print(f"Using metadata directory: {tracker.save_dir}")
            extractor.run()

        self.args.video_root = extractor.save_dir
        self.args.gemini_model_name = "gemini-2.5-pro"  
        if 6 in self.args.stage:
            print("[6/6] Running action & appearance recognition...")
            action_recognizer = GeminiActionRecognizer(
                args=self.args,
                prompt_template=GEMINI_ENTIRE_RECOG_PROMPT, 
                action_prompt_template=GEMINI_SINGLE_ACTION_RECOG_PROMPT
            )
            action_recognizer.run()


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
