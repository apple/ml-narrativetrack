#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

python data_pipeline.py \
    --save_dir run_pipeline \
    --use_owl \
    --owl_model_type base \
    --owl_thres 0.3 \
    --stage 1