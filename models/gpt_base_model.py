# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : gpt_modeling_base.py
#   Last Modified : 2022-04-25 17:17
#   Describe      : 
#
# ====================================================
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from base_model import BaseModel


class GPTBaseModel(BaseModel):
    """
    initiates a PyTorch Lightning GPT2 base model, defines basic training and evaluation steps, offer custom train/valid/test step function for specific tasks
    """
    def __init__(self, args, model=None, tokenizer=None):
        super().__init__(args)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.model = model
        self.tokenizer = tokenizer

    def get_inputs(self, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
        }
        if 'labels' in batch:
            inputs['labels'] = batch['labels']

        return inputs

    def forward(self, input_ids, attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        """ training step """
        inputs = self.get_inputs(batch)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        world_size = self.trainer.world_size
        self._consumed_samples += batch_size * max(world_size, 1)
        labels = inputs["labels"]
        if labels is not None:
            self._consumed_tokens += len(labels.flatten()) * max(world_size, 1)
        else:
            self._consumed_tokens += len(input_ids.flatten()) * max(world_size, 1)

        loss, logits = self(**inputs)

        if labels is not None and self.hparams.show_training_ex > -1 and batch_idx % self.hparams.show_training_ex == 0:
            self.show_training_example(input_ids=input_ids[0], labels=labels[0], logits=logits[0])

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        wandb.log({"train_loss": loss})

        self.ts_logger.add_scalar("train_loss_vs_samples", loss.item(), self._consumed_samples)
        self.ts_logger.add_scalar("train_loss_vs_tokens", loss.item(), self._consumed_tokens)

        output_dict = {"loss": loss}

        return output_dict

    def show_training_example(self, input_ids, labels, logits):
        prediction = torch.argmax(logits, dim=-1)  # (seq_len, vocab_size) -> (seq_len, )
        assert input_ids.size() == labels.size() == prediction.size()  # (seq_len, )
        input_tokens = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        predicted_tokens = self.tokenizer.decode(prediction, skip_special_tokens=True)

        # pad部分被设为了-100，没法decode，所以要把labels==-100变成pad
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels_tokens = self.tokenizer.decode(labels, skip_special_tokens=True)
        print('-' * 50)
        print('input_token:     ', input_tokens)
        print('-' * 50)
        print('predicted_tokens:', predicted_tokens)
        print('-' * 50)
        print('labels_tokens:   ', labels_tokens)
        print('-' * 50)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
