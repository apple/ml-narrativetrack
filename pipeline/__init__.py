#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from pipeline.videollama2 import VideoLLaMA2InferencePipeline
from pipeline.gemini import GeminiVideoPipeline
from pipeline.videochat import VideoChat2InferencePipeline
from pipeline.qwenvl import QwenVLInferencePipeline
from pipeline.internvl import InternVLInferencePipeline
from pipeline.llavavideonext import LLaVAVideoNextInferencePipeline
from pipeline.mplugowl import mPLUGOWL3Pipeline

def get_pipeline(args, device="cuda", schema_type="eval"):
    use_fps = args.use_fps

    if args.model_name == "videollama2":
        args.model_path = "DAMO-NLP-SG/VideoLLaMA2-7B"
        pipeline = VideoLLaMA2InferencePipeline(
            args, 
            device=device,
            fps_1=use_fps,
        )

    elif args.model_name == "videochat2":
        args.model_path = "pretrained/videochat2/videochat2_mistral_7b_stage3.pth"
        args.config_path = "models/VideoChat2/configs/config_mistral.json"
        pipeline = VideoChat2InferencePipeline(
            args,
            device=device,
            fps_1=use_fps,
        )

    elif args.model_name == "llavavideonext":
        args.model_path = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        
        pipeline = LLaVAVideoNextInferencePipeline(
            args,
            device=device,
            fps_1=use_fps,
        )

    elif args.model_name == "qwen":
        args.model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        pipeline = QwenVLInferencePipeline(
            args,
            device=device,
            fps_1=use_fps,
        )
    
    elif args.model_name == "intern":
        args.model_path = "OpenGVLab/InternVL3-8B"
        
        pipeline = InternVLInferencePipeline(
            args,
            device=device,
            fps_1=use_fps,
        )

    elif args.model_name == "mplug_owl3":
        args.model_path = "iic/mPLUG-Owl3-7B-240728"
        pipeline = mPLUGOWL3Pipeline(
            args,
            device=device,
            fps_1=use_fps,
        )

    elif "gemini" in args.model_name:
        pipeline = GeminiVideoPipeline(model_name=args.model_name, schema_type=schema_type)

    else:
        raise ValueError(f"Model {args.model_name} is not supported.")
        
    return pipeline 
