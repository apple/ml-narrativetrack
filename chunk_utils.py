#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
import cv2
from collections import defaultdict

from video_utils import extract_video_timestamps, find_video_file_path
from info import VIDEO_ROOT

# Constants
FPS = 30

def detect_continuous_segments(timestamps, tolerance=1.5):
    """
    Split timestamps into segments where time gaps deviate from expected frame rate continuity.
    A new segment starts if time difference > expected_delta * 1.5
    """
    if not timestamps:
        return []

    segments = []
    start = timestamps[0]
    prev = timestamps[0]

    for ts in timestamps[1:]:
        if ts - prev > tolerance:
            segments.append((start, prev))
            start = ts
        prev = ts

    segments.append((start, prev))
    return segments

def build_entity_trajectories(frames, entity_prefix="entity"):
    trajectories = defaultdict(list)
    for frame in frames:
        frame_idx = frame["frame_idx"]
        for entity in frame.get("entity_tracking", []):
            name = entity["name"]
            if name.startswith(entity_prefix):
                trajectories[name].append({
                    "frame_idx": frame_idx,
                    "timestamp": frame["timestamp"],
                    "bbox": entity["person_bbox"],
                    "actions": entity.get("actions", []),
                    "frame_path": frame.get("frame_path", ""),
                    "metadata": {
                        "scene": frame.get("scene", ""),
                        "entity_detection": entity.get("entity_detection", {}),
                        "entity_tracking": entity.get("entity_tracking", {})
                    }
                })
    return trajectories, len(trajectories)

def overlay_and_save_video(video_type, video_path, metadata, output_path, target_entity="entity"):
    print(f"Overlaying video: {video_path} with metadata for {target_entity}...")
    assert os.path.exists(video_path), f"Video path does not exist: {video_path}"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build frame_idx to bbox mapping
    metadata_by_frame = {m["frame_idx"]: m for m in metadata}          

    frame_idx = 0
    scheduled_boxes = []  # List of dicts with: {'bbox': (x1, y1, x2, y2), 'start': int, 'end': int}

    if video_type == "AVA":
        start_add = 5
        end_add = 15
    elif video_type == "VideoMME":
        if int(fps) > 10:
            start_add = 3
            end_add = int(fps)/2+int(fps)/4
        else:
            start_add = 0
            end_add = 1
    else:
        start_add = 0
        end_add = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Schedule new boxes to appear in future window: frame_idx + 5 to frame_idx + 15
        if frame_idx in metadata_by_frame:
            frame_meta = metadata_by_frame[frame_idx]
            for entity in frame_meta["entity_tracking"]:
                if entity.get("name") == target_entity:
                    bbox = tuple(map(int, entity["person_bbox"]))
                    scheduled_boxes.append({
                        "bbox": bbox,
                        "start": frame_idx + start_add,
                        "end": frame_idx + end_add
                    })

        # Draw boxes only if current frame is in scheduled window
        for box in scheduled_boxes:
            if box["start"] <= frame_idx <= box["end"]:
                x1, y1, x2, y2 = box["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optionally clean up expired boxes
        scheduled_boxes = [b for b in scheduled_boxes if b["end"] >= frame_idx]

        out_vid.write(frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    print(f"Overlayed video saved at: {output_path}")

def get_avg_durations(merged_segments_all):
    """
    Calculate average duration of all segments across all entities.
    Returns the average duration in seconds.
    """
    all_durations = []
    for entity_segments in merged_segments_all.values():
        for start_time, end_time in entity_segments:
            all_durations.append(end_time - start_time)

    avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0
    print(f"Average chunk duration: {avg_duration:.2f} seconds")
    return avg_duration
    
def dynamic_chunk_reappearing(args, save_dir, video_id, metadata, trajectories, min_duration=30, fps=30):
    """
    Identify reappearing segments based on entity trajectories.
    Returns a list of (start_time, end_time) tuples.
    """
    total_chunks = 0
    merged_segments_all = defaultdict(list)
    merged_segments_metadata_all = defaultdict(list)
    offset = 0 if args.video_type in ["TVQA", "TRECVID", "VideoMME", "LVBench"] else 900
    for entity, trajectory in trajectories.items():
        out_dir = os.path.join(save_dir, video_id, entity, "reappear")
        os.makedirs(out_dir, exist_ok=True)
        timestamps = [frame["timestamp"] for frame in trajectory]
        segments = detect_continuous_segments(timestamps)

        i = 0
        n = len(segments)
        
        merged_segments = []
        merged_segments_metadata = {}
        while i < n:
            # Define target_end as 30s from current start
            start_time = segments[i][0]
            target_end_time = start_time + min_duration
            end_time = None

            # Check if any segment fully includes the target_end_time
            for j in range(i, n):
                seg_start, seg_end = segments[j]
                if target_end_time <= seg_end and end_time is None:
                    end_time = seg_end
                    break

            if end_time is None: break
            
            # Collect all segments within (start_time, end_time)
            collected = []
            for j in range(i, n):
                seg_start, seg_end = segments[j]
                if seg_start > end_time: break
                if seg_end >= start_time:
                    collected.append((seg_start, seg_end))

            if len(collected) > 1:
                all_gaps_large = all(
                    collected[k + 1][0] - collected[k][1] > args.reappear_gap_thres
                    for k in range(len(collected) - 1)
                )
                if all_gaps_large:
                    last_start, last_end = collected[-1]
                    merged_segments.append((start_time, end_time))
                    total_chunks += 1
                    output_video_path = extract_video_timestamps(
                        input_video_path=find_video_file_path(video_id, VIDEO_ROOT[args.video_type]),
                        out_dir=out_dir,
                        start_time=start_time,
                        end_time=end_time,
                        offset=offset
                    )

                    # 2. Save metadata
                    adjusted_metadata = []
                    for m in metadata:
                        if start_time <= m["timestamp"] <= end_time:
                            new_m = m.copy()
                            new_m["timestamp"] = m["timestamp"] - start_time  # Adjust timestamp to relative in the subclip
                            new_m["frame_idx"] = m["frame_idx"] - int((start_time-offset) * fps)  # Assuming constant FPS
                            adjusted_metadata.append(new_m)
                    
                    collected = [(s_start-start_time, e_time-start_time) for s_start, e_time in collected]
                    merged_segments_metadata[f"{start_time}-{end_time}"] = {
                        "video_path": output_video_path,
                        "video_segments": collected,
                        "metadata": adjusted_metadata
                    }

                    overlay_video_path = output_video_path.replace(".mp4", "_overlay.mp4")
                    overlay_and_save_video(args.video_type, output_video_path, adjusted_metadata, overlay_video_path, target_entity=entity)

                # Prepare for next round
            if j < n-1:
                i = j
            else:
                break

        merged_segments_all[entity] = merged_segments
        merged_segments_metadata_all[entity] = merged_segments_metadata

    with open(os.path.join(save_dir, video_id, "reappear_metadata.json"), "w") as f:
        json.dump(merged_segments_metadata_all, f, indent=4)

    avg_durations = get_avg_durations(merged_segments_all)
    print("Average duration of reappearing segments:", avg_durations)
    return merged_segments_all, total_chunks, avg_durations

def dynamic_chunk_disappearing(args, save_dir, video_id, metadata, trajectories, min_duration=30, fps=3):
    """
    Identify disappearing segments based on entity trajectories.
    Returns a list of (start_time, end_time) tuples.
    """
    total_chunks = 0
    disappearing_segments_all = defaultdict(list)
    disappearing_segments_metadata_all = defaultdict(list)
    offset = 0 if args.video_type in ["TVQA", "TRECVID", "VideoMME", "LVBench"] else 900
    for entity, trajectory in trajectories.items():
        out_dir = os.path.join(save_dir, video_id, entity, "disappear")
        timestamps = [frame["timestamp"] for frame in trajectory]
        segments = detect_continuous_segments(timestamps)

        start_time = segments[0][0]
        appear_disappear_segments = []
        appear_disappear_segments_metadata = {}
        for i in range(len(segments) - 1):
            seg_start, seg_end = segments[i]
            next_seg_start, _ = segments[i + 1]

            if seg_start + min_duration < next_seg_start and seg_start != seg_end:
                start_time = seg_start
                end_time = min(seg_start + min_duration, next_seg_start - 1)
                appear_disappear_segments.append((start_time, end_time))
                total_chunks += 1
                output_video_path = extract_video_timestamps(
                    input_video_path=find_video_file_path(video_id, VIDEO_ROOT[args.video_type]),
                    out_dir=out_dir,
                    start_time=start_time,
                    end_time=end_time,
                    offset=offset
                )
                # 2. Save metadata
                adjusted_metadata = []
                for m in metadata:
                    if start_time <= m["timestamp"] <= end_time:
                        new_m = m.copy()
                        new_m["timestamp"] = m["timestamp"] - start_time  # Adjust timestamp to relative in the subclip
                        new_m["frame_idx"] = m["frame_idx"] - int((start_time-offset) * fps)  # Assuming constant FPS
                        adjusted_metadata.append(new_m)

                appear_disappear_segments_metadata[f"{start_time}-{end_time}"] = {
                    "video_path": output_video_path,
                    "video_segments": [(start_time-start_time, seg_end-start_time)],
                    "metadata": adjusted_metadata
                }

                overlay_video_path = output_video_path.replace(".mp4", "_overlay.mp4")
                overlay_and_save_video(args.video_type, output_video_path, adjusted_metadata, overlay_video_path, target_entity=entity)
        
        disappearing_segments_all[entity] = appear_disappear_segments
        disappearing_segments_metadata_all[entity] = appear_disappear_segments_metadata

    with open(os.path.join(save_dir, video_id, "disappear_metadata.json"), "w") as f:
        json.dump(disappearing_segments_metadata_all, f, indent=4)
    avg_durations = get_avg_durations(disappearing_segments_all)
    print("Average duration of disappearing segments:", avg_durations)
    return disappearing_segments_all, total_chunks, avg_durations

def dynamic_chunk_appearing(args, save_dir, video_id, metadata, trajectories, min_duration=30, fps=3):
    """
    Identify appearing segments based on entity trajectories.
    Returns a list of (start_time, end_time) tuples.
    """
    total_chunks = 0
    appearing_segments_all = defaultdict(list)
    appearing_segments_metadata_all = defaultdict(list)
    offset = 0 if args.video_type in ["TVQA", "TRECVID", "VideoMME", "LVBench"] else 900
    for entity, trajectory in trajectories.items():
        out_dir = os.path.join(save_dir, video_id, entity, "appear")
        timestamps = [frame["timestamp"] for frame in trajectory]
        segments = detect_continuous_segments(timestamps)

        appearing_segments = []
        appearing_segments_metadata = {}
        for i in range(len(segments) - 1, 0, -1):
            seg_start, seg_end = segments[i]
            prev_seg_end = segments[i - 1][1]

            if seg_end - min_duration > prev_seg_end and seg_end != seg_start:
                end_time = seg_end
                start_time = max(prev_seg_end + 1, seg_end - min_duration)
                appearing_segments.append((start_time, end_time))
                total_chunks += 1
                output_video_path = extract_video_timestamps(
                    input_video_path=find_video_file_path(video_id, VIDEO_ROOT[args.video_type]),
                    out_dir=out_dir,
                    start_time=start_time,
                    end_time=end_time,
                    offset=offset
                )

                # 2. Save metadata
                adjusted_metadata = []
                for m in metadata:
                    if start_time <= m["timestamp"] <= end_time:
                        new_m = m.copy()
                        new_m["timestamp"] = m["timestamp"] - start_time  # Adjust timestamp to relative in the subclip
                        new_m["frame_idx"] = m["frame_idx"] - int((start_time-offset) * fps)  # Assuming constant FPS
                        adjusted_metadata.append(new_m)
                appearing_segments_metadata[f"{start_time}-{end_time}"] = {
                    "video_path": output_video_path,
                    "video_segments": [(seg_start-start_time, seg_end-start_time)],
                    "metadata": adjusted_metadata
                }

                overlay_video_path = output_video_path.replace(".mp4", "_overlay.mp4")
                overlay_and_save_video(args.video_type, output_video_path, adjusted_metadata, overlay_video_path, target_entity=entity)
            
        appearing_segments.reverse()
        appearing_segments_all[entity] = appearing_segments
        appearing_segments_metadata_all[entity] = appearing_segments_metadata

    with open(os.path.join(save_dir, video_id, "appear_metadata.json"), "w") as f:
        json.dump(appearing_segments_metadata_all, f, indent=4)
    avg_durations = get_avg_durations(appearing_segments_all)
    print("Average duration of appearing segments:", avg_durations)
    return appearing_segments_all, total_chunks, avg_durations
