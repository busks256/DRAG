# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule 
from datasets import load_dataset
import torch
import datasets
import transformers
import os
import numpy as np
import copy
import torch.nn as nn
IGNORE_INDEX = -100



class DataModule(LightningDataModule):
    def __init__(self, parser, tokenizer):
        super().__init__()
        self.parser = parser
        self.tokenizer = tokenizer
        emb = np.load("nq_ctx_embeddings.npz")
        ids = emb["ids"]
        vectors = emb["vectors"]
        self.index = {cid: vec for cid, vec in zip(ids, vectors)}
        self.data = datasets.load_dataset("keunha/nq_top20")
        self.prefix_len=10

    def train_dataset_size(self):
        return len(self.data["train"])

    def collate_fn(self, batch):

        input_ids_list = []
        labels_list = []
        attention_masks = []
        ctx_embs_list = []

        for row in batch:
            question = row["question"]
            answer = row["answer"][0]

            # ----------------------------
            # (A) 문서 임베딩 수집
            # ----------------------------
            # shape: (20, embed_dim)
            ctx_embs = [self.index[c["id"]] for c in row["ctxs"]]
            ctx_embs_list.append(ctx_embs)

            # ----------------------------
            # (B) 프롬프트 텍스트 생성
            # ----------------------------
            emb_text = "<EMBED>"*self.prefix_len
            messages = [
                {"role": "system", "content": (
                    "You are an extraction QA assistant. "
                )},
                {
                    "role": "user",
                    "content": (
                        f"Context: {emb_text}\n"
                        f"Question: {question}\n"
                    )
                },{
                    "role": "assistant",
                    "content": (
                        f"The answer is {answer}"
                    )
                }
            ]

            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # ----------------------------
            # (C) tokenization
            # ----------------------------
            encoded = self.tokenizer(
                prompt_text,
                padding=False,
                truncation=False,
                return_tensors=None
            )

            ids = torch.tensor(encoded["input_ids"])

            # <EMBED> 위치 제거

            # labels 생성
            answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

            labels = ids.clone()
            labels[:-len(answer_ids)-2] = -100

            # attention mask
            attention_mask = [1] * len(ids)

            input_ids_list.append(ids )
            labels_list.append(labels)
            attention_masks.append(torch.tensor(attention_mask))

        # ----------------------------
        # (D) padding
        # ----------------------------
        max_len = max(x.size(0) for x in input_ids_list)

        padded_ids = []
        padded_labels = []
        padded_masks = []

        for ids, labels, mask in zip(input_ids_list, labels_list, attention_masks):
            pad = max_len - ids.size(0)
            padded_ids.append(nn.functional.pad(ids, (0, pad), value=self.tokenizer.pad_token_id))
            padded_labels.append(nn.functional.pad(labels, (0, pad), value=-100))
            padded_masks.append(nn.functional.pad(mask, (0, pad), value=0))

        return {
            "input_ids": torch.stack(padded_ids),          # (B, L)
            "attention_mask": torch.stack(padded_masks),   # (B, L)
            "labels": torch.stack(padded_labels),          # (B, L)
            "ctx_embs": torch.tensor(ctx_embs_list),        # (B, 20, embed_dim)
        }
    
    def collate_fn_test(self, batch):

        input_ids_list = []
        labels_list = []
        attention_masks = []
        ctx_embs_list = []
        label = []
        for row in batch:
            question = row["question"]
            answer = row["answer"][0]

            # ----------------------------
            # (A) 문서 임베딩 수집
            # ----------------------------
            # shape: (20, embed_dim)
            ctx_embs = [self.index[c["id"]] for c in row["ctxs"]]
            ctx_embs_list.append(ctx_embs)

            # ----------------------------
            # (B) 프롬프트 텍스트 생성
            # ----------------------------
            emb_text = "<EMBED>"*self.prefix_len
            messages = [
                {"role": "system", "content": (
                    "You are an extraction QA assistant. "
                )},
                {
                    "role": "user",
                    "content": (
                        f"Context: {emb_text}\n"
                        f"Question: {question}\n"
                    )
                }
            ]

            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # ----------------------------
            # (C) tokenization
            # ----------------------------
            encoded = self.tokenizer(
                prompt_text,
                padding=False,
                truncation=False,
                return_tensors=None
            )
        return {
            "input_ids": torch.stack([torch.tensor(encoded["input_ids"])]),          # (B, L)
            "attention_mask": torch.stack([torch.tensor(encoded["attention_mask"])]),  # (B, L)
            "labels" : [answer],
            "ctx_embs": torch.tensor(ctx_embs_list),        # (B, 20, embed_dim)
        }


    def train_dataloader(self):
        return DataLoader(
            self.data["train"],
            batch_size=self.parser.per_gpu_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.parser.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data["validation"],
            batch_size=self.parser.per_gpu_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.parser.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data["validation"].select(range(1000)),  
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn_test,
            num_workers=self.parser.num_workers,
        )