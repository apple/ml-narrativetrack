#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import torchvision.transforms.functional as TF

from video_utils import read_video
from models.VideoChat2.utils.easydict import EasyDict
from models import load_model


class VideoChat2InferencePipeline:
    def __init__(
        self,
        args,
        sys_prompt=None,
        prompt=None,
        device="cuda",
        fps_1=False
    ):
        model, vis_processor, text_processor = load_model(
            "videochat2", test=True, device=device,
            model_path=args.model_path, config_path=args.config_path, num_frames=args.num_frames,
        )
        print(f"Loaded VideoChat2 on {device}")
        self.device = device
        self.model = model
        self.model.eval()
        self.vis_processor = vis_processor
        self.text_processor = text_processor
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
        frames, _, _ = read_video(video_path=video_path, num_frames=self.num_frames, sample="middle")
        video = self.vis_processor(frames)

        if self.save_kframe:
            if ".mp4" in video_id:
                video_id = video_id.replace(".mp4", "")
            for i, frame in enumerate(frames):
                img = TF.to_pil_image(frame)
                os.makedirs(os.path.join(self.out_dir, video_id), exist_ok=True)
                img.save(os.path.join(self.out_dir, video_id, f"frame_{i:03d}.jpg"))

        if input_text is None:
            input_text = self.prompt

        main_prompt, system_prompt = self.format_prompt(input_text, "", self.sys_prompt)

        conversation = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })

        conversation.messages.append([conversation.roles[0], "<Video><VideoHere></Video>\n"])
        if system_prompt is not None:
            prompt_all = system_prompt + main_prompt
        else:
            prompt_all = main_prompt
        conversation = self.text_processor.ask(prompt_all, conversation)

        if len(video.shape) < 5:
            video = video.unsqueeze(0)
        video = video.to(self.device)

        video_emb, _ = self.model.encode_visual_features(video, prompt_all)
        video_list = [video_emb]

        response, _, _ = self.text_processor.answer(
            conv=conversation, model=self.model, video_embs=video_list,
            do_sample=do_sample,
            # temperature=temperature,
            max_new_tokens=max_new_tokens,
            # top_p=top_p,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
        )

        return response
