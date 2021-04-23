#! -*- encoding:utf-8 -*-
"""
@File    :   data.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import random
import json
from random import sample
from typing import Sequence

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# from ptm_task.example import *


class DAPT_Processor(object):
    '''
    args:  dataset_dir, DAPT_version, mask_pct, mask_method, train_batch_size, evltest_batch_size
    '''

    def __init__(self, args, dataset_type, tokenizer) -> None:
        self.args = args
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer

        self.dataset_dir = args.dataset_dir
        self.version = args.DAPT_version
        self.mask_pct = args.mask_pct
        self.max_seq_len = args.max_seq_len
        self.mask_method = args.mask_method  # random
        self.batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size

        self.raw_dapt = None
        self.feature_dict = None
        self.all_concept_tokens = []
        self.all_masked_tokens = []
        self.all_label_tokens = []
        self.all_desc_label = []


    def load_data(self):
        self.load_dapt()
        self.collect()
        self.encode()

    def load_dapt(self):
        dir_dict = {'1.0': 'DAPT_v1'}

        dapt_file = os.path.join(self.dataset_dir, 'dapt', dir_dict[self.version] ,f"DAPT_{self.dataset_type}.json")

        with open(dapt_file, 'r', encoding='utf-8') as f:
            self.raw_dapt = json.load(f)

    def collect(self):

        for case in self.raw_dapt:
            concept = self.tokenizer.tokenize(case['concept'])
            tokens_label = self.tokenizer.tokenize(case['sequence_label'])
            tokens_masked = self.mask_sequence(tokens_label)

            self.all_concept_tokens.append(concept)
            self.all_label_tokens.append(tokens_label)
            self.all_masked_tokens.append(tokens_masked)
            self.all_desc_label.append(0 if case['desc_label'] == "TRUE_DESC" else 1)

    def encode(self):
        self.feature_dict = self.tokenizer.batch_encode_plus(
            list(zip(self.all_masked_tokens, self.all_concept_tokens)), 
            add_special_tokens=True, 
            max_length=self.max_seq_len, 
            padding='max_length', 
            truncation='only_first', 
            return_tensors='pt'
        )
        
        label_feature = self.tokenizer.batch_encode_plus(
            list(zip(self.all_label_tokens, self.all_concept_tokens)), 
            add_special_tokens=True, 
            max_length=self.max_seq_len, 
            padding='max_length', 
            truncation='only_first', 
            return_tensors='pt'
        )['input_ids']

        self.feature_dict['sequence_labels'] = label_feature
        self.feature_dict['desc_labels'] = torch.tensor(self.all_desc_label).long()

    def mask_sequence(self, token_list):
        tokens_masked = token_list[::]

        indices = [i for i in range(len(token_list))]
        mask_indices = random.choices(indices, k=int(self.mask_pct*len(token_list)))
        for index in mask_indices:
            tokens_masked[index] = self.tokenizer.mask_token
        return tokens_masked

    def make_dataloader(self, shuffle=True):
        order = ['input_ids', 'attention_mask', 'token_type_ids', 'sequence_labels', 'desc_labels']
        data = [self.feature_dict[key] for key in order]

        dataset = TensorDataset(*data)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, drop_last=False)

        return dataloader


if __name__ == "__main__":
    import argparse
    from transformers import BertTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--DAPT_version', type=str)
    parser.add_argument('--mask_method', type=str, choices=['random'])
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--mask_pct', type=float, default=0.15)
    parser.add_argument('--train_batch_size', type=int, default=0)
    parser.add_argument('--evltest_batch_size', type=int, default=8)

    args_str = r"""
        --dataset_dir D:\CODE\Commonsense\CSQA_dev\DATA\
        --DAPT_version 1.0
        --mask_method random
        --mask_pct 0.15
        --max_seq_len 40
        --evltest_batch_size 16
        --train_batch_size 8
        """
    args = parser.parse_args(args_str.split())

    tokenizer = BertTokenizer.from_pretrained(r"D:\CODE\Python\Transformers-Models\bert-base-cased")

    processor = DAPT_Processor(args, 'dev', tokenizer)
    processor.load_data()
    dataloader = processor.make_dataloader()
    for batch in dataloader:
        print(batch[-1])
