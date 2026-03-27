#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import math

import numpy as np
from PIL import Image
from decord import VideoReader, cpu

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class QwenVLInferencePipeline:
    def __init__(
        self,
        args,
        sys_prompt=None,
        prompt=None,
        device="cuda",
        fps_1=False
    ):
        self.device = device
        self.num_frames = args.num_frames
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.save_kframe = args.save_kframe

        self.num_frames = args.num_frames
        MIN_PIX = 256 * 28 * 28     # ~256 tokens/image
        MAX_PIX =  640 * 28 * 28    # keep modest; raise to 896/1280 if you have more VRAM

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=MIN_PIX, max_pixels=MAX_PIX)

    def get_video_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])
        return video_path, frames, timestamps

    def create_image_grid(self, images, num_columns=8):
        pil_images = [Image.fromarray(image) for image in images]
        num_rows = math.ceil(len(images) / num_columns)

        img_width, img_height = pil_images[0].size
        grid_width = num_columns * img_width
        grid_height = num_rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height))

        for idx, image in enumerate(pil_images):
            row_idx = idx // num_columns
            col_idx = idx % num_columns
            position = (col_idx * img_width, row_idx * img_height)
            grid_image.paste(image, position)

        return grid_image

    def generate_response(
        self,
        input_text, video_path, video_id, q_type=None,
        max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):

        frames = self.get_video_frames(video_path)[1]  # (video_path, frames, timestamps)
        pil_frames = [Image.fromarray(f) for f in frames]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": input_text},
                # Instead of a full video, give a list of image frames:
                *[{"type": "image", "image": pil} for pil in pil_frames]
            ]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=pil_frames,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

