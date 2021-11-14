# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : TianHongZXY
#   Email         : tianhongzxy@163.com
#   File Name     : bert_baseline.py
#   Last Modified : 2021-11-13 00:58
#   Describe      : 
#
# ====================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from models.base_model import BaseModel, MLPLayer, OutputLayer, Pooler
from torchsnooper import snoop
from ChildTuningOptimizer import ChildTuningAdamW
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap


class Bert(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('Bert')
        
        # * Args for pretrained model setting
        parser.add_argument('--pretrained_model_name',
                            default='bert-base-cased',
                            type=str)
        parser.add_argument('--child_tuning', action='store_true', default=False, help="if use the child tuning optimizer")
        parser.add_argument('--finetune', action='store_true', default=False, help="if fine tune the pretrained model")
        parser.add_argument("--pooler_type", type=str, default="cls", help="acceptable values:[cls, cls_before_pooler, avg, avg_top2, avg_first_last]")
        parser.add_argument('--bert_lr', default=1e-5, type=float)
        parser.add_argument('--bert_l2', default=0., type=float)
        parser.add_argument('--mlp_dropout', default=0.5, type=float, help="Dropout rate in MLP layer")

        return parent_args

    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.hidden_size = self.config.hidden_size
        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.bert.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        if not self.hparams.finetune:
            for name, child in self.bert.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        self.init_model(args)

    def adv_forward(self, logits, train_inputs):
        adv_loss = self.adv_loss_fn(model=self,
                                    logits=logits,
                                    train_inputs=train_inputs
                                    )

        adv_loss = self.hparams.adv_alpha * adv_loss

        return adv_loss

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                            inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)

        pooler_output = self._pooler(attention_mask, outputs)
        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        logits = self.output(pooler_output)
        logits = logits.view(-1, self.nlabels)

        return logits

    def predict(self, input_ids=None, attention_mask=None, token_type_ids=None):
        logits = self(input_ids=input_ids, 
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids, 
                      )

        predict = logits.argmax(dim=-1)
        predict = predict.cpu().tolist()

        return predict

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_paras = list(self.bert.named_parameters())
        bert_paras = [
            {'params': [p for n, p in bert_paras if not any(nd in n for nd in no_decay)], 'weight_decay': self.hparams.bert_l2, 'lr': self.hparams.bert_lr},
            {'params': [p for n, p in bert_paras if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.hparams.bert_lr}
        ]

        named_paras = list(self.named_parameters())
        head_paras = [
            {'params': [p for n, p in named_paras if 'bert' not in n], 'lr': self.hparams.lr}
        ]
        #  print('head_paras:', head_paras)

        paras = bert_paras + head_paras

        if self.hparams.child_tuning:
            optimizer = ChildTuningAdamW(paras, lr=self.hparams.lr)
        else:
            optimizer = AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.total_step * self.hparams.warmup), self.total_step)

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]

    def init_model(self, args):
        """
        init function.
        """
        self.pooler_type = args.pooler_type
        self._pooler = Pooler(args.pooler_type)
        if args.pooler_type == "cls":
            self.mlp = MLPLayer(self.hidden_size, args.mlp_dropout)
        self.output = OutputLayer(self.hidden_size, self.nlabels)

    #  def configure_sharded_model(self):
    #      self.mlp = auto_wrap(self.mlp)
    #      self.output = auto_wrap(self.output)
    #      self._pooler = auto_wrap(self._pooler)
    #      #  self.bert = auto_wrap(self.bert)
    #      self.model = nn.Sequential(self.mlp, self.output, self._pooler, self.bert)


