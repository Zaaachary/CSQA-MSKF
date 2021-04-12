#! -*- encoding:utf-8 -*-
"""
@File    :   data_processor.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import json
from random import sample

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from csqa_task.example import *


class Baseline_Processor(object):
    
    def __init__(self, args, dataset_type):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = dataset_type
        self.data = None

    def load_data(self):
        data = []

        # load raw data
        f = open(os.path.join(self.dataset_dir, 'csqa', f"{self.dataset_type}_rand_split.jsonl"), 'r', encoding='utf-8')
        for line in f:
            data.append(line.strip())
        f.close()

        # convert raw data 2 CSQAexample
        for index, case in enumerate(data):
            case = json.loads(case)
            example = CSQAExample.load_from_json(case)
            data[index] = example

        self.data = data

    def make_dataloader(self, tokenizer, args, shuffle=True):
        batch_size = args.train_batch_size if self.dataset_type == 'train' else args.evltest_batch_size
        drop_last = False

        T, L = [], []

        for example in tqdm(self.data):
            text_list, label = example.tokenize(tokenizer, args)
            T.append(text_list)
            L.append(label)
        
        self.data = (T, L)  # len(T) = len(L)
        return self._convert_to_tensor(batch_size, drop_last, shuffle)

    def _convert_to_tensor(self, batch_size, drop_last, shuffle):
        tensors = []

        features = self.data[0]     # tensor, label
        all_input_ids = torch.tensor([[f.input_ids for f in fs] for fs in features], dtype=torch.long)
        all_input_mask = torch.tensor([[f.input_mask for f in fs] for fs in features], dtype=torch.long)
        all_segment_ids = torch.tensor([[f.segment_ids for f in fs] for fs in features], dtype=torch.long)

        # features
        # tensors.extend((all_input_ids, all_input_mask, all_segment_ids))
        tensors.extend((all_input_ids, all_input_mask, all_segment_ids))
        
        # labels
        tensors.append(torch.tensor(self.data[1], dtype=torch.long))
        # b, 5, len; b,

        dataset = TensorDataset(*tensors)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        return dataloader


class OMCS_Processor(object):
    '''
    add multi cs at the end of the sequence.
    cs_num, max_seq_len
    '''
    
    def __init__(self, args, dataset_type):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = dataset_type
        self.version = args.OMCS_version
        self.raw_data = []
        self.examples = []
        self.omcs_cropus = None
        self.omcs_index = None

    def load_data(self):
        self.load_csqa()    # csqa dataset
        if self.version == '1':
            self.load_omcs()    # omcs free text
            self.load_csqa_omcs_index() # csqa -ESindex-> omcs
            self.inject_commonsense()
        elif self.version[0] == '2':
            self.load_omcsv2()
            self.inject_commonsensev2()

    def load_csqa(self):
        f = open(os.path.join(self.args.dataset_dir, 'csqa', f"{self.dataset_type}_rand_split.jsonl"), 'r', encoding='utf-8')
        for line in f:
            self.raw_data.append(json.loads(line.strip()))
        f.close()

    def load_omcs(self):
        omcs_file = os.path.join(self.dataset_dir, 'omcs', "omcs_v1","omcs-free-origin.json")
        with open(omcs_file, 'r', encoding='utf-8') as f:
            self.omcs_cropus = json.load(f)
    
    def load_csqa_omcs_index(self):
        cs_result_file = os.path.join(self.dataset_dir, 'omcs', "omcs_v1", f'{self.dataset_type}_QAconcept-Match_omcs_of_dataset.json')
        with open(cs_result_file, 'r', encoding='utf-8') as f:
            self.omcs_index = json.load(f)
    
    @staticmethod
    def load_example(case, cs4choice):
        return OMCSExample.load_from(case, cs4choice)

    def inject_commonsense(self):
        '''
        put commonsense into case, accroding to omcs_index (ES result)

        '''
        for case in self.raw_data:
            cs4choice = {}  # {choice: csforchoice}
            choice_csindex = self.omcs_index[case['id']]['endings']
            for cs_index in choice_csindex:
                # cs for single choice, choose top self.args.cs_num
                cs_list = list(map(int, cs_index['cs'][:self.args.cs_num]))
                cs_list = [self.omcs_cropus[cs] for cs in cs_list]

                cs_list.sort(key=lambda x:len(x)) # sort by cs_len

                # some case don't have cs_num cs
                temp = self.args.cs_num - len(cs_list)
                if temp:
                    cs_list.extend(['<unk>']*temp)

                cs4choice[cs_index['ending']] = cs_list

            example = self.load_example(case, cs4choice)
            self.examples.append(example)

    def load_omcsv2(self):
        dir_dict = {'2.2':'omcs_v2.2_15', '2.3':'omcs_v2.3_10'}

        omcs_file = os.path.join(self.dataset_dir, 'omcs', dir_dict[self.version] ,f"{self.dataset_type}_rand_split_omcs.json")

        with open(omcs_file, 'r', encoding='utf-8') as f:
            self.omcs_cropus = json.load(f)

    def inject_commonsensev2(self):
        omcs_index = 0
        for case in self.raw_data:
            cs4choice = {}
            for choice in case['question']['choices']:
                choice_test = choice['text']
                cs_list = self.omcs_cropus[omcs_index]['cs_list'][:self.args.cs_num]
                omcs_index += 1

                cs_list.sort(key=lambda x:len(x)) # sort by cs_len

                temp = self.args.cs_num - len(cs_list)
                if temp:
                    cs_list.extend(['<unk>']*temp)

                cs4choice[choice_test] = cs_list
            
            example = self.load_example(case, cs4choice)
            self.examples.append(example)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size
        drop_last = False

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []

        # debug = -1
        for example in tqdm(self.examples):
            # debug += 1
            # if debug == 1457:
            #     import pdb; pdb.set_trace()
            # call example's tokenize function
            # feature_dict: [5, 128], [5, 128], [5, 128]
            # import pdb; pdb.set_trace()
            feature_dict = example.tokenize(tokenizer, args)
            all_input_ids.append(feature_dict['input_ids'])
            all_token_type_ids.append(feature_dict['token_type_ids'])
            all_attention_mask.append(feature_dict['attention_mask'])
            all_label.append(example.label)
        
        all_input_ids = torch.stack(all_input_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        all_token_type_ids = torch.stack(all_token_type_ids)
        all_label = torch.tensor(all_label, dtype=torch.long)

        data = (all_input_ids, all_attention_mask, all_token_type_ids, all_label)

        dataset = TensorDataset(*data)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        return dataloader
        

class CSLinear_Processor(OMCS_Processor):
    '''
    Base on OMCS_Processor, add restriction to the sequence len.
    question_len, cs_len, cs_num, max_qa_len 54, max_cs_len 18
    '''
    
    def __init__(self, args, dataset_type):
        super().__init__(args, dataset_type)
        self.max_qa_len = args.max_qa_len
        self.max_cs_len = args.max_cs_len
        self.max_seq_len = args.max_seq_len
    
    @staticmethod
    def load_example(case, cs4choice):
        # import pdb; pdb.set_trace()
        return CSLinearExample.load_from(case, cs4choice)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        return super().make_dataloader(tokenizer, args, shuffle=shuffle)


class MultiSource_Processor(OMCS_Processor):

    def __init__(self, args, dataset_type):
        super().__init__(args, dataset_type)
