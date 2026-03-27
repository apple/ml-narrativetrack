#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

def load_model(model, *args, **kwargs):
    """
    Returns a triple of (model, vis_processor, text_processor). If your model does not require any of these, you may return None
    """
    # Lazy load models, due to different requirements
    if model == "videollama2":
        from models import VideoLLaMA2
        return VideoLLaMA2.load_model(*args, **kwargs)
    elif model == "videochat2":
        from models import VideoChat2
        return VideoChat2.load_model(*args, **kwargs)
    else:
        return {
            "random" : (None, None, None),
            "gpt-4o" : lambda *x, **y : ("gpt-4o", None, None),
            "gemini-1.5-pro" : lambda *x, **y : ("gemini-1.5-pro", None, None),
            "gemini-1.5-flash" : lambda *x, **y : ("gemini-1.5-flash", None, None)
        }[model](*args, **kwargs)
