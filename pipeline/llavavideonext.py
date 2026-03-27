#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import torchvision.transforms.functional as TF
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor


class LLaVAVideoNextInferencePipeline:
    def __init__(
        self,
        args,
        sys_prompt=None,
        prompt=None,
        device="cuda",
        fps_1=False
    ):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        )
        processor = LlavaNextVideoProcessor.from_pretrained(args.model_path)

        self.model = model.to(device)
        self.device = device
        self.model.eval().to(self.device)
        self.processor = processor
        self.num_frames = args.num_frames
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.save_kframe = args.save_kframe

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(
        self, input_text, video_path, video_id,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=128,
        num_return_sequences=1,
        num_beams=1,
        q_type=None,
    ):
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "video", "path": video_path},
                    ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation, num_frames=self.num_frames, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        response = response.split("ASSISTANT:")[-1].strip()
        return response
