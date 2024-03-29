import os
import json
from random import random, sample
from copy import deepcopy
import logging

from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from csqa_task.data import ProcessorBase
from rank_task.example import OMCSrankExample


class RankOMCS_Processor(ProcessorBase):
    '''
    CSRK_version
    split_method
    '''
    def __init__(self, args, dataset_type):
        super(RankOMCS_Processor, self).__init__(args, dataset_type)
        self.csrk_version = args.CSRK_version
        self.split_method = args.split_method
        self.csqa_cs_list = []

    def load_data(self):
        if self.args.mission == "train":
            self.load_dataset()
            self.inject_omcs_rank()

    def load_dataset(self):
        dir_dict = {
            '0.1': 'csrk_v0.1',
            '0.2': 'csrk_v0.2',
            '0.3': 'csrk_v0.3',
            }

        f = open(os.path.join(self.args.dataset_dir, 'csrk', dir_dict[self.csrk_version], f"{self.dataset_type}_csrank.json"), 'r', encoding='utf-8')
        self.csqa_cs_list = json.load(f)
        f.close()
        
    def inject_omcs_rank(self):
        if self.dataset_type == 'dev':
            self.csqa_cs_list = random.sample(self.csqa_cs_list, k=500)
        else:
            self.csqa_cs_list = random.sample(self.csqa_cs_list, k=10000)

        for choice_case in self.csqa_cs_list:

            if self.split_method == 'half':
                if len(choice_case['cs_list']) < 2:
                    continue
                half = int(len(choice_case['cs_list'])/2)
                front = choice_case['cs_list'][:half]
                back = choice_case['cs_list'][half:]
            
            elif self.split_method == 'topbotton2':
                if len(choice_case['cs_list']) >= 4:
                    front = choice_case['cs_list'][:2]
                    back = choice_case['cs_list'][-2:]
                elif len(choice_case['cs_list']) >=2:
                    front = choice_case['cs_list'][:1]
                    back = choice_case['cs_list'][-1:]
                else:
                    continue

            example = OMCSrankExample.load_from(choice_case, front, choice_case['isanswer'])
            self.examples.append(example)
            example = OMCSrankExample.load_from(choice_case, back, not choice_case['isanswer'])
            self.examples.append(example)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        drop_last = False

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []

        seq_len = args.max_seq_len

        for example in tqdm(self.examples):
            feature_dict, labels = example.tokenize(tokenizer, args)
            all_input_ids.extend(feature_dict['input_ids'])
            all_token_type_ids.extend(feature_dict['token_type_ids'])
            all_attention_mask.extend(feature_dict['attention_mask'])
            all_label.extend(labels)


        all_input_ids = torch.tensor(all_input_ids)
        all_attention_mask = torch.tensor(all_attention_mask)
        all_token_type_ids = torch.tensor(all_token_type_ids)

        all_label = torch.tensor(all_label, dtype=torch.long)

        data = (all_input_ids, all_attention_mask, all_token_type_ids, all_label)

        dataset = TensorDataset(*data)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, drop_last=drop_last)

        return dataloader