# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : dataloader.py
#   Last Modified : 2021-11-13 00:58
#   Describe      : 
#
# ====================================================

import time
import argparse
import itertools
import json
import copy
import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning import Trainer, seed_everything, loggers
from models.bert_baseline import Bert
from torchsnooper import snoop
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TaskDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--data_dir',
                            default='./data',
                            type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--cached_train_data',
                            default='cached_train_data.pkl',
                            type=str)
        parser.add_argument('--cached_valid_data',
                            default='cached_valid_data.pkl',
                            type=str)
        parser.add_argument('--cached_test_data',
                            default='cached_test_data.pkl',
                            type=str)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=32, type=int)
        parser.add_argument('--recreate_dataset', action='store_true', default=False)
        
        return parent_args
    
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers
        self.pretrained_model = args.pretrained_model
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize

        self.cached_data_dir = os.path.join(args.data_dir, args.pretrained_model_name)
        if not os.path.exists(self.cached_data_dir):
            os.makedirs(self.cached_data_dir)

        self.cached_train_data_path = os.path.join(self.cached_data_dir, args.cached_train_data)
        self.cached_valid_data_path = os.path.join(self.cached_data_dir, args.cached_valid_data)
        self.cached_test_data_path = os.path.join(self.cached_data_dir, args.cached_test_data)

        self.train_data_path = os.path.join(args.data_dir, args.train_data)
        self.valid_data_path = os.path.join(args.data_dir, args.valid_data)
        self.test_data_path = os.path.join(args.data_dir, args.test_data)

        # Whether to recreate dataset, useful when using a new pretrained model with different tokenizer, 
        # Default false, reuse cached data if exist
        self.recreate_dataset = args.recreate_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_data = self.create_dataset(self.cached_train_data_path,
                                                 self.train_data_path)
            self.valid_data = self.create_dataset(self.cached_valid_data_path,
                                                 self.valid_data_path)
        if stage == 'test':
            self.test_data = self.create_dataset(self.cached_test_data_path,
                                                self.test_data_path,
                                                test=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, \
            batch_size=self.train_batchsize, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def create_dataset(self, cached_data_path, data_path, test=False):
        if  os.path.exists(cached_data_path) and not self.recreate_dataset:
            print(f'Loading cached dataset from {cached_data_path}...')
            data = torch.load(cached_data_path)
            #  Filter data if you don't need all of them
            #  data = list(filter(lambda x: len(self.acronym2lf[x['acronym']]) < 15 and (x['acronym'] in self.ori_diction or random.random() < 0.2), data))
            output = f'Load {len(data)} instances from {cached_data_path}.'
        else:
            print(f'Preprocess {data_path} for TASK NAME...')
            dataset = json.load(open(data_path, 'r'))
            data = []

            for example in tqdm(dataset):
                sentence = example['sentence']
                # Do not return_tensors here, otherwise rnn.pad_sequence in collate_fn will raise error
                encoded = self.tokenizer(sentence, truncation=True, max_length=512)
                encoded['sentence'] = sentence
                encoded['input_ids'] = torch.LongTensor(encoded['input_ids'])
                encoded['attention_mask'] = torch.LongTensor(encoded['attention_mask'])
                encoded['token_type_ids'] = torch.LongTensor(encoded['token_type_ids'])
                #  for ids in encoded["input_ids"]:
                #      print(tokenizer.decode(ids))

                if not test:
                    label = int(example['label'])
                    encoded['label'] = label

                #  Customize your example here if needed
                #  input_ids = encoded['input_ids']
                #  attention_mask = encoded['attention_mask']
                #  Models like roberta don't have token_type_ids
                #  if 'token_type_ids' not in encoded:
                #      encoded['token_type_ids'] = [[0] * len(x) for x in input_ids]
                #  example = {
                #      'sentence': sentence,
                #      'input_ids': torch.LongTensor(input_ids),
                #      'attention_mask': torch.LongTensor(attention_mask),
                #      'token_type_ids': torch.LongTensor(encoded['token_type_ids']),
                #  }

                data.append(encoded)

            output = f'Load {len(data)} instances from {data_path}.'
            data = TaskDataset(data)
            torch.save(data, cached_data_path)
            print('Last example:', encoded)

        print(output)
        return data

    def collate_fn(self, batch):
        '''
        Aggregate a batch data.
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {'sentence':[ins1_sentence, ins2_sentence...], 'input_ids':[ins1_input_ids, ins2_input_ids...], ...}
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        token_type_ids = batch_data['token_type_ids']
        labels = None
        if 'label' in batch_data:
            labels = torch.LongTensor(batch_data['label'])

        # Before pad input_ids = [tensor<seq1_len>, tensor<seq2_len>, ...]
        # After pad input_ids = tensor<batch_size, max_seq_len>
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                              batch_first=True,
                                              padding_value=self.tokenizer.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                   batch_first=True,
                                                   padding_value=0)
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                   batch_first=True,
                                                   padding_value=0)

        batch_data = {
            'sentence': batch_data['sentence'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
        }

        return batch_data


class TaskDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser()

    # * Args for data preprocessing
    total_parser = Task2DataModel.add_data_specific_args(total_parser)
    
    # * Args for training
    #  total_parser = Trainer.add_argparse_args(total_parser)

    # * Args for model specific
    total_parser = Bert.add_model_specific_args(total_parser)

    args = total_parser.parse_args()


    # * Here, we test the data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                              use_fast=True)

    task2_data = Task2DataModel(args, tokenizer)

    task2_data.setup('fit')
    task2_data.setup('test')

    val_dataloader = task2_data.val_dataloader()

    batch = next(iter(val_dataloader))

    print(batch)

