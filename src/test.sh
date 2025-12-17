#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py --devices 1 --do_test_only --ckpt_path "../checkpoint/None/epoch=2-val_loss=1.4617-v1.ckpt" --per_gpu_batch_size 1 --num_workers 0
