# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : train.py
#   Last Modified : 2021-11-13 00:58
#   Describe      : 
#
# ====================================================

import os
import json
import torch
import argparse
from itertools import chain
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AdamW
from dataloader import TaskDataset, TaskDataModel
from models.base_model import BaseModel
from models.bert_baseline import Bert
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks.progress import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    # Set global random seed
    seed_everything(args.seed)
    # Set path to save checkpoint and outputs
    if args.checkpoint_path is not None:
        save_path = os.path.split(args.checkpoint_path)[0]
        args.save_path = save_path
    else:
        hyparas = 'bs={}-lr={}-pooler={}-model={}-l2={}-ft={}-clip={}-drop={}-adv={}-prec-{}'.format(
                    args.train_batchsize, args.lr, args.pooler_type, args.pretrained_model_name, args.l2,
                    int(args.finetune), args.gradient_clip_val, args.mlp_dropout, int(args.adv), args.precision)
        save_path = os.path.join(args.save_dir, args.model_name)
        save_path = os.path.join(save_path, hyparas)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.save_path = save_path

        # Prepare Trainer
        checkpoint = ModelCheckpoint(dirpath=save_path,
                                     save_top_k=3,
                                     save_last=True,
                                     monitor='valid_acc',
                                     mode='max',
                                     filename='{epoch:02d}-{valid_acc:.4f}')
        checkpoint.CHECKPOINT_NAME_LAST = "{epoch}-last"
        early_stop = EarlyStopping(monitor='valid_acc',
                                   mode='max',
                                   patience=5,
                                   check_on_train_epoch_end=True) # Check early stopping after every train epoch, ignore multi validation in one train epoch
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(save_path, 'logs/'), name=hyparas)
        trainer = Trainer.from_argparse_args(args, 
                                             logger=logger,
                                             callbacks=[checkpoint, early_stop])
        # Set path to load pretrained model
        args.pretrained_model = os.path.join(args.pretrained_model_dir, args.pretrained_model_name)

        # Save args
        with open(os.path.join(save_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        print(k, ":", v, end=',\t')
    print('\n' + '-' * 64)

    # Set tokenizer and data model 
    if args.eval:
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        data_model = TaskDataModel(args, tokenizer)
        checkpoint_path = args.checkpoint_path
    # Training
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                                  use_fast=True)
        tokenizer.save_pretrained(save_path)
        data_model = TaskDataModel(args, tokenizer)
        model = Bert(args, tokenizer)
        trainer.fit(model, data_model)
        checkpoint_path = checkpoint.best_model_path

    # Evaluation
    print("Load checkpoint from {}".format(checkpoint_path))
    model = Bert.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)

    # Evaluating on dev set
    evaluation(args, model, data_model, output_save_path, mode='fit')
    # Evaluating on test set
    evaluation(args, model, data_model, output_save_path, mode='test')

def evaluation(args, model, data_model, save_path, mode):
    data_model.setup(mode)
    tokenizer = data_model.tokenizer
    if mode == 'fit':
        test_loader = data_model.val_dataloader()
    else:
        test_loader = data_model.test_dataloader()
    model.cuda()
    model.eval()

    results = []
    for batch in tqdm(test_loader):
        predicts = model.predict(batch['input_ids'].cuda(),
                                 batch['attention_mask'].cuda(),
                                 batch['token_type_ids'].cuda(),
                                 )

        for idx, predict in enumerate(predicts):
            pred = {
                'sentence': batch['sentence'][idx],
                'label': predict
            }
            results.append(pred)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluation file saved at {}".format(save_path))


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument("--pretrained_model_dir", default="/home/zxy21/codes_and_data/.cache/pretrained_models/", 
                            type=str, help="Path to the directory which contains all the pretrained models downloaded from huggingface")

    # * Args for data preprocessing
    total_parser = TaskDataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    # * Args for base model 
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * Args for base specific model 
    total_parser = Bert.add_model_specific_args(total_parser)

    args = total_parser.parse_args()
    torch.set_num_threads(args.num_threads)
    
    main(args)

