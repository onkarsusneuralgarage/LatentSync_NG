#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=8000 -m scripts.train_syncnet \
    --config_path "configs/syncnet/syncnet_16_pixel_attn.yaml"
