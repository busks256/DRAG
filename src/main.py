# -*- coding: utf-8 -*-

import os, random, time
import wandb
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer
from config import *
from datamodule import *
from model import TKG_LLM   

def main(parser):
    
    if parser.wandb == False:
        os.environ["WANDB_MODE"] = "offline"
    
    seed_everything(parser.seed)
    
    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(parser.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens_dict = {"additional_special_tokens": ["<EMBED>"]}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print("Added special tokens:", num_added)
    # datamodule
    datamodule = DataModule(parser, tokenizer)
    train_loader, val_loader, test_loader = datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()

    run_name = parser.run_name

    # wandb.login(key=parser.wandb_key)
    # wandb_logger = WandbLogger(
    #         name = run_name,
    #         project = "nq-sum",
    #         entity = 'nlplab-skku',  
    #         config=vars(parser)
    #     )

    if parser.load_from is None:
        model= TKG_LLM(parser)
    else:
        print(f'loading from {parser.load_from}...')
        model = TKG_LLM.load_from_checkpoint(parser.load_from)
    checkpoint_callback = ModelCheckpoint(
        dirpath = f'../checkpoint/{run_name}',
        filename = '{epoch}-{val_loss:.4f}',
        monitor = 'val_loss',
        save_top_k = 3,
        mode = 'min'
    )
    
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 3,
        mode = 'min'
    )
    
    trainer = Trainer(
        accelerator = 'gpu',
        devices = parser.devices, 
        strategy = "ddp",
        precision='16-mixed',
        # logger = wandb_logger,
        max_epochs = parser.epochs,
        accumulate_grad_batches = parser.gradient_accumulation_steps,
        callbacks = [checkpoint_callback, early_stopping],
    )
    
    if not parser.do_test_only:
        trainer.fit(model, train_loader, val_loader)

    if not parser.do_train_only:
        if parser.ckpt_path is not None:
            trainer.test(model, test_loader, ckpt_path = parser.ckpt_path)
        else:
            trainer.test(model, test_loader, ckpt_path = 'best')
    wandb.finish()

if __name__ == '__main__':
    parser = get_options()
    main(parser)      
    
