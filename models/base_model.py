# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : base_model.py
#   Last Modified : 2021-11-13 00:58
#   Describe      : 
#
# ====================================================


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsnooper import snoop
from adversarial_loss import AdversarialLoss
import time


class Pooler(pl.LightningModule):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        #  use this if pretrained_model.forward(return_dict=True)
        #  last_hidden = outputs.last_hidden_state
        #  hidden_states = outputs.hidden_states
        last_hidden = outputs[0]
        hidden_states = outputs[2]

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class MLPLayer(pl.LightningModule):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    Same way as CLIP-Adapter, except dropout is added
    """

    def __init__(self, hidden_size, alpha=0.5, dropout_rate=0.5):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout_rate)
        self.proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                   nn.ReLU(), 
                                   nn.Linear(hidden_size, hidden_size),
                                   )

    def forward(self, features):
        x = self.alpha * self.proj(features) + (1 - self.alpha) * features
        x = self.dropout(x)

        return x


class OutputLayer(pl.LightningModule):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, features):
        logits = self.proj(features)

        return logits


class BaseModel(pl.LightningModule):
    """
    Define training_step, validation_step, optimizer and other general training setting here
    """
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        
        # * Args for general setting
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--checkpoint_path', default=None, type=str)
        parser.add_argument('--seed', default=20020206, type=int)
        parser.add_argument('--save_dir', default='./save', type=str)
        parser.add_argument('--model_name', default='ModelName', type=str)
        parser.add_argument('--lr', default=1e-5, type=float)
        parser.add_argument('--l2', default=0., type=float)
        parser.add_argument('--warmup', default=0.1, type=float)
        parser.add_argument('--adv', action='store_true', default=False, help="if use adversarial training")
        parser.add_argument('--divergence', default='js', type=str)
        parser.add_argument('--adv_nloop', default=1, type=int,
                            help="1 (default), inner loop for getting the best perturbations.")
        parser.add_argument('--adv_step_size', default=1e-3, type=float,
                            help="1 (default), perturbation size for adversarial training.")
        parser.add_argument('--adv_alpha', default=1, type=float,
                            help="1 (default), trade off parameter for adversarial training.")
        parser.add_argument('--noise_var', default=1e-5, type=float)
        parser.add_argument('--noise_gamma', default=1e-6, type=float, help="1e-4 (default), eps for adversarial copy training.")
        parser.add_argument('--project_norm_type', default='inf', type=str)

        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()

        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.args = args
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.adv = args.adv
        if args.adv:
            self.adv_loss_fn = AdversarialLoss(args)
        # Set number of different labels here or add it to args
        self.nlabels = 10

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.total_step = int(self.trainer.max_epochs * len(train_loader) / \
                (max(1, self.trainer.gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def train_inputs(self, batch):
        #  Filter reduntant information(for example: 'sentence') that will be passed to model.forward()
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids']
        }
        return inputs 

    def training_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        labels = batch['labels']
        logits = self(**inputs)

        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))

        if self.adv:
            adv_loss = self.adv_forward(logits=logits, train_inputs=inputs)

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == batch['labels']).long().sum()
        acc = ncorrect / ntotal

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        if self.adv:
            self.log('adv_loss', adv_loss, on_step=True, prog_bar=True)
            return loss + adv_loss

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        labels = batch['labels']
        logits = self(**inputs)

        predict = logits.argmax(dim=-1).cpu().tolist()

        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == batch['labels']).long().sum()
        acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_step=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=True, prog_bar=True)

        return ncorrect.detach().cpu(), ntotal, predict, labels.cpu().tolist()

    def validation_epoch_end(self, validation_step_outputs):
        ncorrect = 0
        ntotal = 0
        predictions = []
        labels = []
        for x in validation_step_outputs:
            ncorrect += x[0]
            ntotal += x[1]
            predictions += x[2]
            labels += x[3]

        recall = recall_score(labels, predictions, average="macro")
        precision = precision_score(labels, predictions, average="macro")
        f1 = f1_score(labels, predictions, average='macro')
        self.log('valid_acc_epoch', ncorrect / ntotal, on_epoch=True, prog_bar=True)
        self.log('valid_recall', recall, on_epoch=True, prog_bar=True)
        self.log('valid_precision', precision, on_epoch=True, prog_bar=True)
        self.log('valid_f1', f1, on_epoch=True, prog_bar=True)

        print("ncorrect = {}, ntotal = {}".format(ncorrect, ntotal))
        print(f"Validation Accuracy: {round(ncorrect / ntotal, 3)}")
        print('Validation P: {:.3%}, R: {:.3%}, F1: {:.3%}'.format(precision, recall, f1))

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.l2
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.hparams.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

