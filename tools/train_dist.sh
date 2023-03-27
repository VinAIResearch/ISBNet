#!/usr/bin/env bash
GPUS=$1

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/scannetv2/isbnet_scannetv2.yaml --trainall --exp_name default