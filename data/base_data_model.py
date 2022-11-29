# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : base_data_model.py
#   Last Modified : 2022-04-13 15:28
#   Describe      : 
#
# ====================================================
import os
import logging
from functools import partial
from typing import List, Union, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import coloredlogs
logger = logging.getLogger('__file__')
coloredlogs.install(level='INFO', logger=logger)


class BaseDataset(Dataset):
    def __init__(self, data: Union[List, Tuple]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class BaseDataModel(pl.LightningDataModule):
    """ PyTorch Lightning data class """
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('BaseDataModel')
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default=None, type=str)
        parser.add_argument('--valid_data', default=None, type=str)
        parser.add_argument('--test_data', default=None, type=str)
        parser.add_argument('--predict_data', default=None, type=str)
        parser.add_argument('--micro_batch_size', default=4, type=int)
        parser.add_argument('--valid_batch_size', default=4, type=int)
        parser.add_argument('--test_batch_size', default=4, type=int)
        parser.add_argument('--predict_batch_size', default=4, type=int)
        parser.add_argument('--source_max_token_len', default=None, type=int)
        parser.add_argument('--target_max_token_len', default=None, type=int)
        parser.add_argument('--recreate_dataset', action='store_true', default=False)
        parser.add_argument('--task', default=None, type=str)
        parser.add_argument('--data_name', default=None, type=str)

        return parent_parser

    def __init__(self, args, tokenizer):
        """
        Initiates a PyTorch Lightning Data Model
        """
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer

        if self.hparams.task is not None:
            self.hparams.data_dir = os.path.join(self.hparams.data_dir, self.hparams.task)
        else:
            logger.warning("You didn't specifiy the task name, using default cached data path."
                           "This may cause problems when you using the same dataset for different tasks.")
            self.hparams.data_dir = os.path.join(self.hparams.data_dir, "undefined_task")
        self.cached_data_root_path = os.path.join(self.hparams.data_dir, os.path.split(self.hparams.model_name)[-1])
        if not os.path.exists(self.cached_data_root_path):
            os.makedirs(self.cached_data_root_path)

    def prepare_data(self):
        if self.hparams.train_data:
            self.create_data_cache(path=os.path.join(self.hparams.data_dir, self.hparams.train_data), type='train')
        if self.hparams.valid_data:
            self.create_data_cache(path=os.path.join(self.hparams.data_dir, self.hparams.valid_data), type='valid')
        if self.hparams.test_data:
            self.create_data_cache(path=os.path.join(self.hparams.data_dir, self.hparams.test_data), type='test')
        if self.hparams.predict_data:
            self.create_data_cache(path=os.path.join(self.hparams.data_dir, self.hparams.predict_data), type='predict')

    def setup(self, stage):
        self.raw_train_data = self.load_data_cache(type='train')
        if self.hparams.valid_data:
            self.raw_valid_data = self.load_data_cache(type='valid')
        if self.hparams.test_data:
            self.raw_test_data = self.load_data_cache(type='test')
        if self.hparams.predict_data:
            self.raw_predict_data = self.load_data_cache(type='predict')

    def get_examples(self, path) -> List:
        '''Load raw data into list from files'''
        raise NotImplementedError()

    @staticmethod
    def collate_fn(batch, args, tokenizer) -> Dict:
        '''Puts each data field into a tensor with outer dimension batch size'''
        raise NotImplementedError()

    def create_data_cache(self, path, type):
        '''Load raw data from cache if exists, otherwise load and then create cache'''
        cached_data_path = os.path.join(self.cached_data_root_path, f'{type}_cached.pkl')
        if os.path.exists(cached_data_path) and not self.hparams.recreate_dataset:
            print(f'Cached data already exists in {cached_data_path}, skip creating.')
            return

        print(f'Preprocess data from {path}, save in {cached_data_path}...')
        data = self.get_examples(path)
        torch.save(data, cached_data_path)

    def load_data_cache(self, type) -> List:
        cached_data_path = os.path.join(self.cached_data_root_path, f'{type}_cached.pkl')
        if not os.path.exists(cached_data_path):
            raise ValueError(f"Cached data doesn't exist in {cached_data_path}!")

        print(f'Loading cached data from {cached_data_path}...')
        data = torch.load(cached_data_path)
        print(f"Load {len(data)} {type} examples.")

        return data

    def train_dataloader(self):
        return DataLoader(
            BaseDataset(self.raw_train_data),
            batch_size=self.hparams.micro_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.collate_fn, args=self.hparams, tokenizer=self.tokenizer),
        )

    def val_dataloader(self):
        if self.hparams.valid_data:
            return DataLoader(
                BaseDataset(self.raw_valid_data),
                batch_size=self.hparams.valid_batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=partial(self.collate_fn, args=self.hparams, tokenizer=self.tokenizer),
            )

        return None

    def test_dataloader(self):
        if self.hparams.test_data:
            return DataLoader(
                BaseDataset(self.raw_test_data),
                batch_size=self.hparams.test_batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=partial(self.collate_fn, args=self.hparams, tokenizer=self.tokenizer),
            )

        return None

    def predict_dataloader(self):
        if self.hparams.predict_data:
            return DataLoader(
                BaseDataset(self.raw_predict_data),
                batch_size=self.hparams.predict_batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=partial(self.collate_fn, args=self.hparams, tokenizer=self.tokenizer),
            )

        return None
