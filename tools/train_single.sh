#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py configs/scannetv2/isbnet_scannetv2.yaml --trainall --exp_name default