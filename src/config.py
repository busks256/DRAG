# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--warmup_ratio', type=float, default=0.3)
        self.parser.add_argument('--data_names', type=str , nargs='+', choices=['dolly', 'COT', 'everything',  'law', 'num', ], help="dolly | everyLM | CoT | law")
        self.parser.add_argument('--data_dir', type=str, default='../data/', help='data directory')
        self.parser.add_argument('--load_from', type=str, default=None, help='when do continual learning, where to load trained model')
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--run_name', type=str, default=None)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')
        self.parser.add_argument('--lr_scheduler_step', type=bool, default=False)
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', 
                        help='save dataset with cross-attention scores')


        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoint/', help='models are saved here')
        # self.parser.add_argument('--data_dir', type=str, default='./data/', help='models are saved here')
        # self.parser.add_argument('--data_dir', type=str, default='../data/', help='data directory')
        self.parser.add_argument('--base_model', type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='pretrained model name')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--devices', type=int, default=1, help='devices for training')
        self.parser.add_argument('--wandb', type=bool, default=False, help='wandb')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=16, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)
        self.parser.add_argument("--max_seq_length", default=256, type=int, 
                        help="max sequence length.")
        self.parser.add_argument("--local-rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        self.parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
        self.parser.add_argument('--n_shot', type=int, default=0, help='number of shots for few-shot learning')

        # training parameters
        self.parser.add_argument('--eval_freq', type=int, default=50,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=5000,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=50,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')
        self.parser.add_argument('--epochs', type=int, default=5,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient accumulation steps')
        self.parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
        self.parser.add_argument("--lora_r", type=int, default=8)
        self.parser.add_argument("--lora_alpha", type=int, default=32)
        self.parser.add_argument("--lora_dropout", type=float, default=0.1)
        self.parser.add_argument("--do_train_only", action="store_true", help="only train the model")
        self.parser.add_argument("--do_test_only", action="store_true", help="only test the model")
        self.parser.add_argument("--wandb_key", type=str, default="330f5e7f972f801a6a619738f037a401d797469d", help="wandb key")
        self.parser.add_argument("--dataset_name", type=str, default="GEM/web_nlg", help="dataset name in the datasets library")
        
        
        
        
        self.parser.add_argument('--ckpt_path', type=str, default=None, help='path to save the checkpoints')
    def parse(self):
        opt = self.parser.parse_args()
        return opt


def get_options():
    options = Options()
    return options.parse()
