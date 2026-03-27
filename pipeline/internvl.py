#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoProcessor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVLInferencePipeline:
    def __init__(
        self,
        args,
        sys_prompt=None,
        prompt=None,
        device="cuda",
        fps_1=False
    ):
        self.fps_1 = fps_1
        self.device = device
        self.num_frames = args.num_frames
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.save_kframe = args.save_kframe
        self.num_frames = args.num_frames

        self.model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)


    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def get_indices_fixed_fps(self, fps, max_frame, fps_out=1.0, bound=None, first_idx=0):
        """
        Return frame indices sampled at `fps_out` (e.g., 1 FPS), independent of num_frames.
        `bound` is in seconds: (start_sec, end_sec). If None, use full video.
        """
        # derive time range in seconds
        if bound is not None:
            start_sec, end_sec = float(bound[0]), float(bound[1])
        else:
            start_sec, end_sec = 0.0, max_frame / float(fps)

        # clamp to valid range
        start_sec = max(0.0, start_sec)
        end_sec   = min(end_sec, max_frame / float(fps))
        if end_sec <= start_sec:
            return np.array([], dtype=int)

        # timestamps at fixed rate, inclusive end (tiny epsilon)
        ts = np.arange(start_sec, end_sec + 1e-9, 1.0 / float(fps_out))
        # map timestamps to nearest frame indices
        idx = np.round(ts * float(fps)).astype(int)
        idx = np.clip(idx, first_idx, max_frame)
        # remove duplicates (can happen if fps is low / rounding)
        idx = np.unique(idx)
        return idx

    def load_video(
        self, video_path, 
        bound=None, input_size=448, max_num=1, fps_out=None):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        if fps_out is not None:
            frame_indices = self.get_indices_fixed_fps(fps, max_frame, fps_out=fps_out, bound=bound, first_idx=0)
        else:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=self.num_frames)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def generate_response(
        self, input_text, video_path, video_id, q_type=None, 
        bound=None, input_size=448, max_num=1, do_sample=True, max_new_tokens=2048
    ):
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if self.fps_1:
            pixel_values, num_patches_list = self.load_video(video_path, max_num=1, fps_out=1.0)
        else:
            pixel_values, num_patches_list = self.load_video(video_path, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + input_text
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        return response
