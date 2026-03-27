#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
from decord import VideoReader
import io
import subprocess
import cv2
import json
import os
from PIL import Image
import random
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from info import VIDEO_ROOT

"""
Adapted from VideoChat2: https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/dataset/video_utils.py
"""
def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, fps=1, max_num_frames=-1, clip=None):
    if sample in ["rand", "middle"]: # Uniform sampling
        acc_samples = min(num_frames, vlen)
        # Split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            if clip:
                start_idx, end_idx = round(clip[0] * fps), min(round(clip[1] * fps), max_num_frames)
            else:
                if max_num_frames < 0:
                    max_num_frames = vlen - 1
                start_idx, end_idx  = 0, max_num_frames
            
            seg_size = float(end_idx - start_idx) / num_frames
            frame_indices = np.array([
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_frames)
            ])
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # Sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / fps
        delta = 1 / output_fps  # Gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def get_all_videos(video_type, face_ref_dir, stage=1, dark_video_filter=True, filtered_video_path=None):
    video_root = VIDEO_ROOT[video_type]
    video_candidates = [
        os.path.join(root, f)
        for root, _, files in os.walk(video_root)
        for f in files
        if not (f.lower().endswith(('.txt', '.DS_Store')) or 'tar.gz' in f.lower())
    ]
    print("len(video_candidates): ", len(video_candidates))
    if dark_video_filter:
        print(filtered_video_path)
        assert filtered_video_path is not None, "Filtered video path must be provided when dark_video_filter is True."
        filtered_video_path = os.path.join(filtered_video_path, f"{video_type}_non_dark_final_list.txt")
        assert os.path.exists(filtered_video_path)
        with open(filtered_video_path) as f:
            filtered_videos = set(line.strip() for line in f if line.strip())

    print(f"Total videos found in {video_root}: {len(video_candidates)}")
    video_paths = []
    for video_path in video_candidates:
        if video_type == "AVA":
            video_id = Path(video_path).stem
            if dark_video_filter:
                if Path(video_path).name in filtered_videos:
                    if stage > 2: 
                        assert os.path.exists(os.path.join(face_ref_dir, video_type, video_id)), \
                            f"Face reference directory does not exist for {video_id}."
                        video_paths.append(video_path)
                    else:
                        video_paths.append(video_path)
            else:
                video_paths.append(video_path)
        else:
            if video_path.endswith(".mp4") or video_path.endswith(".mov"):
                video_id = "/".join(video_path.split("/")[-2:])
                video_id = video_id.replace(".mp4", "")
                video_id = video_id.replace(".mov", "")
                if dark_video_filter:
                    if "/".join(video_path.split("/")[-2:]) in filtered_videos:
                        if stage > 2: 
                            tmp_path = os.path.join(face_ref_dir, video_type, video_id)
                            if os.path.exists(tmp_path) and len(os.listdir(tmp_path))!= 0:
                                video_paths.append(video_path)
                        else:
                            video_paths.append(video_path)
                else:
                    video_paths.append(video_path)
    return video_paths

def find_video_file_path(video_id, base_dir="datasets/ava/videos"):
    if 'tvqa' in base_dir.lower() or 'videomme' in base_dir.lower() or 'lvbench' in base_dir.lower():
        return os.path.join(base_dir, f"{video_id}.mp4")
    else:
        matching_files = [f for f in os.listdir(base_dir) if f.startswith(video_id)]
        assert len(matching_files) == 1, f"Expected exactly one video file for {video_id}, found {len(matching_files)}"
        return os.path.join(base_dir, matching_files[0])

def extract_video_timestamps(input_video_path, out_dir, start_time, end_time, offset=900):
    """
    Extracts a subclip using ffmpeg and saves to: video_id/entity_id/clip_type/clip.mp4

    Args:
        input_video_path: str, full path to the source video file
        out_dir: str, directory to save the extracted clip
        start_time: float, in seconds
        end_time: float, in seconds
    """
    duration = end_time - start_time + 1
    if duration <= 0:
        print(f"Skipping invalid clip: {start_time}-{end_time}")
        return

    # Build output path
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{int(start_time)}_{int(end_time)}.mp4")

    cmd = [
        "ffmpeg",
        "-i", input_video_path,
        "-ss", str(start_time-offset),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_file
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved clip: {output_file} -- {start_time}s to {end_time}s")
    return output_file
    
def get_video_resolution(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    probe = json.loads(result.stdout)
    width = probe['streams'][0]['width']
    height = probe['streams'][0]['height']
    return width, height

def load_frame_fn(video_root, video_id, timestamp):
    """
    Load frame as PIL.Image from video_id at timestamp (in seconds).
    
    Args:
        video_id: e.g. 'abcd1234'
        timestamp: float, in seconds
        video_root: directory where videos are stored
    
    Returns:
        frame_img: PIL.Image
    """
    video_path = find_video_file_path(video_id, video_root)
    assert os.path.exists(video_path), f"Video file {video_path} does not exist."
    # Use ffmpeg to extract frame at timestamp
    command = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # Get video resolution → can parse ffprobe if needed; here hardcode for 854x480 AVA (default)
    w, h = get_video_resolution(video_path)
    
    raw_frame = result.stdout
    if len(raw_frame) != w * h * 3:
        raise RuntimeError(f"Failed to load frame for video {video_id} at {timestamp}s.")
    
    frame_img = Image.frombytes('RGB', (w, h), raw_frame)
    return frame_img

def read_video(
    video_path, num_frames=None, sample='rand', fix_start=None, client=None, clip=None, reverse=False
):
    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if num_frames is None:
        num_frames = round(duration)

    # Hack: To get all frames
    if num_frames == -1:
        num_frames = vlen

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start, fps=fps, max_num_frames=vlen - 1, clip=clip
    )
    frames = video_reader.get_batch(frame_indices)
    if not isinstance(frames, torch.Tensor):
        frames = torch.tensor(frames.asnumpy())
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    if reverse:
        frames = frames.flip(0)              # reverse along time dimension
        frame_indices = frame_indices[::-1]  # keep indices consistent

    return frames, frame_indices, float(fps)

def save_clip_as_video(frames, save_path, fps=30):
    """
    Save a list of frames (BGR) as a video file.

    Args:
        frames (List[np.ndarray]): List of BGR frames (np.array of shape [H, W, 3])
        save_path (str): Output .mp4 file path
        fps (int): Frames per second
    """
    if not frames:
        print("[WARN] No frames to write.")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1'
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"[INFO] Saved video: {save_path}")

def load_all_frames(video_path, flip=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if flip: frame = cv2.rotate(frame, cv2.ROTATE_180)
        frames.append(frame)
    cap.release()
    fps = int(fps)
    return frames, fps

def load_ava_annotations(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']
    return df

def get_timestamps_for_video(ava_df, video_id):
    """
    Returns sorted list of unique AVA timestamps (int) for the given video_id.
    """
    timestamps = ava_df[ava_df['video_id'] == video_id]['timestamp'].unique()
    return sorted(timestamps)

def load_frame_from_images(args, frame_root_dir, video_id, timestamp, fps=30, start_sec=900):
    frame_idx = (timestamp - start_sec) * fps + 1  # +1 since ffmpeg uses 1-based indexing
    if args.video_type == "AVA":
        frame_filename = f"{video_id}_{frame_idx:06d}.jpg"
    else:
        frame_filename = f"{frame_idx:05d}.jpg"
    frame_path = os.path.join(frame_root_dir, video_id, frame_filename)

    if not os.path.isfile(frame_path):
        print(f"[WARN] Frame not found: {frame_path}")
        return None, None, frame_path

    img = Image.open(frame_path)
    return img, img.size, frame_path  # img.size = (width, height)

def extract_bboxes_for_timestamp(ava_df, video_name, timestamp, img_size):
    """
    Returns list of dicts, one per person_id, with merged action_ids and pixel bbox.
    """
    width, height = img_size
    rows = ava_df[(ava_df['video_id'] == video_name) & (ava_df['timestamp'] == timestamp)]

    grouped = {}
    for _, row in rows.iterrows():
        person_id = int(row['person_id'])
        x1 = int(row['x1'] * width)
        y1 = int(row['y1'] * height)
        x2 = int(row['x2'] * width)
        y2 = int(row['y2'] * height)
        key = (person_id, x1, y1, x2, y2)

        if key not in grouped:
            grouped[key] = {
                "person_id": person_id,
                "bbox": [x1, y1, x2, y2],
                "action_ids": []
            }
        grouped[key]["action_ids"].append(int(row['action_id']))

    return list(grouped.values())
