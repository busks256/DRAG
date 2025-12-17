import json
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import torch.optim as optim
from torch.optim import AdamW, Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer, AutoModelForCausalLM , get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


import pdb
import gzip
import pickle
import json




class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        modules = [nn.Linear(2560, 2560)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(2560,2560))
        self.projector = nn.Sequential(*modules)
    
    def forward(self,context_embedding):
        return self.projector(context_embedding)


class TKG_LLM(LightningModule):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser 
        self.tokenizer = AutoTokenizer.from_pretrained(self.parser.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(self.parser.base_model)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        lora_config = LoraConfig(
                r=parser.lora_r,
                lora_alpha=parser.lora_alpha,
                lora_dropout=parser.lora_dropout,
                target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"],  # 모델 구조 확인 필요
                bias="none",
                task_type="CAUSAL_LM",
            )
        self.model = get_peft_model(base_model, lora_config)
        print("[LoRA] Enabled with rank={}, alpha={}, dropout={}".format(
            parser.lora_r, parser.lora_alpha, parser.lora_dropout
        ))
        self.embedding_layer = self.model.get_input_embeddings()
        self.proj = Projector()
        self.validation_step_outputs = []
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, label):
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = label
        )
        return outputs
    
    def generate(self, *args, **kwargs):
        return self.model.generate(
            *args, **kwargs
        )
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), betas=(0.9, 0.999), eps=1e-8, lr = self.parser.lr, weight_decay=self.parser.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches*self.parser.warmup_ratio,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
		"scheduler": scheduler,
		"interval": "step",
	    }
        return  [optimizer],[scheduler_config]
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['labels']
        prompt = batch['ctx_embs']
        prompt_position = torch.where(input_ids==151669)
        input_ids[prompt_position]=0
        input_embeds = self.embedding_layer(input_ids)
        input_prompt = self.proj(prompt)
        input_embeds = torch.cat([input_embeds[:,:28,:],input_prompt,input_embeds[:,38:]],dim=1)
        
        outputs = self.model(
            inputs_embeds = input_embeds,
            attention_mask = attention_mask,
            labels = label
        )
        
        loss = outputs['loss']
        if torch.isnan(loss).item() is True:
            print(self.tokenizer.decode(input_ids[0]))
            import pdb; pdb.set_trace()
            return None
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, prog_bar=True, sync_dist=True,logger=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True,logger=True)
        return {
            'loss' : loss,
            'log': {'train_loss': loss, 'learning_rate': lr}
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['labels']
        prompt = batch['ctx_embs']
        prompt_position = torch.where(input_ids==151669)
        input_ids[prompt_position]=0
        input_embeds = self.embedding_layer(input_ids)
        input_prompt = self.proj(prompt)
        input_embeds = torch.cat([input_embeds[:,:28,:],input_prompt,input_embeds[:,38:]],dim=1)
        
        outputs = self.model(
            inputs_embeds = input_embeds,
            attention_mask = attention_mask,
            labels = label
        )
        
        loss = outputs['loss']
        self.validation_step_outputs.append(loss)
        return {
            'loss' : loss
        }


    def on_validation_epoch_end(self):
        val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log('val_loss', val_loss, prog_bar = True, sync_dist=True)
    def on_test_epoch_start(self):
        self.test_step_outputs = []
        self.test_step_labels = []


    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        prompt = batch['ctx_embs']
        # Replace prompt placeholder (151669 → 0)
        prompt_position = torch.where(input_ids == 151669)
        input_ids[prompt_position] = 0

        # Embedding
        input_embeds = self.embedding_layer(input_ids)
        input_prompt = self.proj(prompt)

        # Insert prompt segment
        input_embeds = torch.cat([
            input_embeds[:, :28, :],
            input_prompt,
            input_embeds[:, 38:, :]
        ], dim=1)

        # Generate prediction
        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=32
        )

        # ----- FIX: sequence 단위로 저장 -----
        self.test_step_outputs.append(self.tokenizer.decode(outputs[0][:-1]))


    def on_test_epoch_end(self):
        
        output_path = "test_outputs.json"
        import pdb; pdb.set_trace()
        results = [
            {"pred_short_answer": p} for p in self.test_step_outputs
        ]

        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Saved predictions to {output_path}")