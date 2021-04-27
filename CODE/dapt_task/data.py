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


class Webster_Processor(object):
    '''
    args:  dataset_dir, DAPT_version, mask_pct, mask_method, train_batch_size, evltest_batch_size
    '''

    def __init__(self, args, dataset_type, tokenizer) -> None:
        self.args = args
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer

        self.dataset_dir = args.dataset_dir
        self.version = args.Webster_version
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


class OMCS_Processor:

    def __init__(self, args, dataset_type, tokenizer) -> None:
        self.args = args
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer

        self.dataset_dir = args.dataset_dir
        self.mask_pct = args.mask_pct
        self.max_seq_len = args.max_seq_len
        self.mask_method = args.mask_method  # random
        self.batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size

        self.omcs_cropus = None
        self.all_label_tokens = []
        self.all_masked_tokens = []

    def load_data(self):
        self.load_omcs()
        self.mask_token()
        self.encode()

    def load_omcs(self):
        omcs_file = os.path.join(self.dataset_dir, 'omcs', 'omcs_dapt' ,f"{self.dataset_type}_omcs.json")
        f = open(omcs_file, 'r', encoding='utf-8')
        self.omcs_cropus = json.load(f)
        f.close()

    def mask_token(self):
        for case in self.omcs_cropus:
            tokens_label = self.tokenizer.tokenize(case)
            tokens_masked = self.mask_sequence(tokens_label)

            self.all_label_tokens.append(self.tokenizer.convert_tokens_to_ids(tokens_label))
            self.all_masked_tokens.append(self.tokenizer.convert_tokens_to_ids(tokens_masked))

    def encode(self):
        all_input_ids, all_token_type_ids, all_attention_mask, all_sequencs_label = [], [], [], []

        for masked_tokens in self.all_masked_tokens:
            feature_dict = self.tokenizer.encode_plus(
                masked_tokens, 
                add_special_tokens=True, 
                max_length=self.max_seq_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            all_input_ids.append(feature_dict['input_ids'].squeeze(0))
            all_token_type_ids.append(feature_dict['token_type_ids'].squeeze(0))
            all_attention_mask.append(feature_dict['attention_mask'].squeeze(0))

            label_feature = self.tokenizer.encode_plus(
                masked_tokens, 
                add_special_tokens=True, 
                max_length=self.max_seq_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )['input_ids'].squeeze(0)

            all_sequencs_label.append(label_feature)

        self.feature_dict = {
            'input_ids': torch.stack(all_input_ids),
            'token_type_ids': torch.stack(all_token_type_ids),
            'attention_mask': torch.stack(all_attention_mask),
            'sequence_labels': torch.stack(all_sequencs_label),
        }

    def mask_sequence(self, token_list):
        tokens_masked = token_list[::]

        indices = [i for i in range(len(token_list))]
        mask_indices = random.choices(indices, k=int(self.mask_pct*len(token_list)))
        for index in mask_indices:
            tokens_masked[index] = self.tokenizer.mask_token
        return tokens_masked
    
    def make_dataloader(self, shuffle=True):
        order = ['input_ids', 'attention_mask', 'token_type_ids', 'sequence_labels']
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
    parser.add_argument('--Webster_version', type=str)
    parser.add_argument('--mask_method', type=str, choices=['random'])
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--mask_pct', type=float, default=0.15)
    parser.add_argument('--train_batch_size', type=int, default=0)
    parser.add_argument('--evltest_batch_size', type=int, default=8)

    args_str = r"""
        --dataset_dir D:\CODE\Commonsense\CSQA_dev\DATA\
        --Webster_version 1.0
        --mask_method random
        --mask_pct 0.15
        --max_seq_len 40
        --evltest_batch_size 16
        --train_batch_size 8
        """
    args = parser.parse_args(args_str.split())

    tokenizer = BertTokenizer.from_pretrained(r"D:\CODE\Python\Transformers-Models\bert-base-cased")

    processor = Webster_Processor(args, 'dev', tokenizer)
    processor.load_data()
    dataloader = processor.make_dataloader()
    for batch in dataloader:
        print(batch[0].shape)

    processor = OMCS_Processor(args, 'dev', tokenizer)
    processor.load_data()
    dataloader = processor.make_dataloader()
    for batch in dataloader:
        print(batch[-1].shape)
