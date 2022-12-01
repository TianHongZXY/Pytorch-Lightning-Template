# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : gpt_base_data_model.py
#   Last Modified : 2022-11-30 17:36
#   Describe      : 
#
# ====================================================
import os
from base_data_model import BaseDataModel
from data_preprocess import DataProcessor
from typing import List, Dict


class GPTBaseDataModel(BaseDataModel):
    def __init__(self, args, tokenizer): 
        super().__init__(args, tokenizer)

    def get_examples(self, path) -> List:
        '''Load raw data into list from files'''
        file_type = os.path.splitext(path)[-1].replace(".", "")
        examples = getattr(DataProcessor, f"_read_{file_type}")
        assert isinstance(examples, list)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer) -> Dict:
        '''
        Here we suppose each example is a dict.
        Puts each data field into a tensor with outer dimension batch size.
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        inputs = tokenizer(
            batch_data['text'], 
            add_special_tokens=True, 
            padding=True,
            truncation=True,
            max_length=args.source_max_token_len, 
            return_tensors="pt",
            return_attention_mask=True, 
            )

        return dict(**batch_data, **inputs)


if __name__ == '__main__':
    import argparse
    from transformers import GPT2Tokenizer

    total_parser = argparse.ArgumentParser()
    # * data preprocessing args
    total_parser = BaseDataModel.add_data_specific_args(total_parser)
    args = total_parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_data_model = GPTBaseDataModel(args, tokenizer)

    train_dataloader = gpt_data_model.train_dataloader()
    print(len(gpt_data_model.raw_train_data))

    batch = next(iter(train_dataloader))
    print(batch)

