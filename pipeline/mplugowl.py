#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import torch
from modelscope import AutoModel, AutoTokenizer

class mPLUGOWL3Pipeline:
    def __init__(
        self,
        args,
        sys_prompt=None,
        prompt=None,
        device="cuda",
        fps_1=False
    ):
        self.device = device
        self.fps_1 = fps_1
        self.num_frames = args.num_frames
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.save_kframe = args.save_kframe
        self.num_frames = args.num_frames

        self.model  = AutoModel.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True)
        _ = self.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.processor = self.model.init_processor(self.tokenizer)

    def encode_video(self, video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > self.num_frames:
            frame_idx = uniform_sample(frame_idx, self.num_frames)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames


    def encode_video_fps(self, video_path, fps_out=1.0, bound=None):
        """
        Read video frames at a fixed output FPS (default 1 FPS).
        - bound: optional (start_sec, end_sec). If None, uses the full video.
        - If more than self.num_frames frames are gathered, uniformly downsample.
        Returns: List[PIL.Image]
        """

        vr = VideoReader(video_path, ctx=cpu(0))
        n_total = len(vr)
        if n_total == 0:
            return []

        src_fps = float(vr.get_avg_fps()) or 30.0
        # derive time window
        if bound is not None:
            start_sec = max(0.0, float(bound[0]))
            end_sec   = min(float(bound[1]), (n_total - 1) / src_fps)
        else:
            start_sec = 0.0
            end_sec   = (n_total - 1) / src_fps

        if end_sec <= start_sec:
            return []

        # timestamps at fixed rate (e.g., 0s, 1s, 2s, ...)
        ts = np.arange(start_sec, end_sec + 1e-9, 1.0 / float(fps_out))
        # map timestamps -> nearest frame index
        frame_idx = np.round(ts * src_fps).astype(int)
        frame_idx = np.clip(frame_idx, 0, n_total - 1)
        # dedupe and keep ascending order
        frame_idx = np.unique(frame_idx)

        # fetch frames
        frames_np = vr.get_batch(frame_idx.tolist()).asnumpy()
        frames = [Image.fromarray(f.astype("uint8")).convert("RGB") for f in frames_np]
        return frames

    def generate_response(
        self,
        input_text, video_path, video_id, q_type=None,
    ):
        
        messages = [
            {"role": "user", "content": f"""<|video|>{input_text}"""},
            {"role": "assistant", "content": ""}
        ]
        if self.fps_1:
            video_frames = [self.encode_video_fps(video_path, fps_out=1.0)]
        else:
            video_frames = [self.encode_video(video_path)] 
        inputs = self.processor(messages, images=None, videos=video_frames).to(self.device)
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 100,
            'decode_text': True,
        })

        output_text = self.model.generate(**inputs)[0]
        return output_text
