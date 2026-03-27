#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
import os
import cv2
import numpy as np
from models import load_model
from models.VideoLLaMA2.utils.mm_utils import tokenizer_multimodal_token, KeywordsStoppingCriteria
from video_utils import read_video
import torchvision.transforms.functional as TF

def read_video_at_fps(video_path, fps_out=1.0, start_sec=0.0, end_sec=None, as_rgb=True):
    """
    Return frames sampled at a fixed output rate (e.g., 1 FPS), independent of num_frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_total / src_fps if src_fps > 0 else 0.0

    if end_sec is None or end_sec > duration:
        end_sec = duration

    # timestamps we want to sample at: 0s, 1s, 2s, ...
    ts = np.arange(start_sec, end_sec + 1e-9, 1.0 / fps_out)

    frames, used_ts = [], []
    for t in ts:
        # map timestamp to nearest frame index
        idx = int(round(t * src_fps))
        if idx >= n_total: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: break
        if as_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        used_ts.append(t)

    cap.release()
    return frames, used_ts, {"src_fps": src_fps, "duration": duration}

class VideoLLaMA2InferencePipeline:
    def __init__(
        self,
        args,
        sys_prompt=None,
        prompt=None,
        device="cuda",
        fps_1=False
    ):
        model, vis_processor, text_processor = load_model(
            "videollama2", device=device,
            model_path=args.model_path, config_path=args.config_path,
            num_frames=args.num_frames, load_4bit=args.load_4bit, load_8=args.load_8bit,
            # LLaVa-NeXT-Video override parameters
            mm_spatial_pool_mode=args.mm_spatial_pool_mode, 
            mm_newline_position=args.mm_newline_position,
            mm_pooling_position=args.mm_pooling_position,
        )
        self.fps_1 = fps_1
        self.model = model.to(device)
        self.model = model
        self.device = device
        self.model.eval()
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.num_frames = args.num_frames
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.save_kframe = args.save_kframe

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return self.text_processor.prepare_prompt(f"{main_prompt}\n\n{options_prompt}"), system_prompt
    
    def generate_response(
        self, input_text, video_path, video_id, 
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=512,
        num_return_sequences=1,
        num_beams=1,
        q_type=None,
    ):
        
        if self.fps_1:
            frames, _, _ = read_video_at_fps(video_path, fps_out=1.0)
        else:
            frames, _, _ = read_video(video_path=video_path, num_frames=self.num_frames, sample="middle")
        print(len(frames))
        video = self.vis_processor(frames)
        if self.save_kframe:
            if '.mp4' in video_id: video_id = video_id.replace('.mp4', '')
            for i, frame in enumerate(frames):
                img = TF.to_pil_image(frame)  
                os.makedirs(os.path.join(self.out_dir, video_id), exist_ok=True)
                img.save(os.path.join(self.out_dir, video_id, f"frame_{i:03d}.jpg"))

        if input_text is None:
            input_text = self.prompt
        main_prompt, system_prompt = self.format_prompt(input_text, "", self.sys_prompt)
        input_ids = tokenizer_multimodal_token(
            main_prompt, self.text_processor.tokenizer, self.text_processor.visual_token, return_tensors="pt"
        ).unsqueeze(0).long().to(self.device)
        attention_masks = input_ids.ne(self.text_processor.tokenizer.pad_token_id).long().to(self.device)

        stopping_criteria = KeywordsStoppingCriteria([
            self.text_processor.tokenizer.eos_token
        ], self.text_processor.tokenizer, input_ids)
        
        with torch.no_grad(), torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=[(video.half().to(self.device), self.text_processor.modality)],
                do_sample=do_sample,
                # temperature=temperature,
                max_new_tokens=max_new_tokens,
                # top_p=top_p,
                # use_cache=True,
                stopping_criteria=[stopping_criteria],
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                pad_token_id=self.text_processor.tokenizer.eos_token_id,
            )
        
        outputs = self.text_processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = outputs[0].strip() if len(outputs) <= 1 else [x.strip() for x in outputs]
        return outputs
