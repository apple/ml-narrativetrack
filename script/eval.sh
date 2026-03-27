#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

python eval_pipeline.py \
    --num_gpus 1 \
    --procs_per_gpu 1 \
    --model_name videollama2 \
    --qa_path datasets/narrativetrack_qa.json