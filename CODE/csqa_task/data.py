#! -*- encoding:utf-8 -*-
"""
@File    :   data_processor.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import json
from random import random, sample
from copy import deepcopy
import logging

logger = logging.getLogger("data processor")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from csqa_task.example import *


class ProcessorBase(object):

    def __init__(self, args, dataset_type) -> None:
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = dataset_type
        self.batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size
        self.raw_csqa = []
        self.examples = []

    def load_data(self):
        # override
        pass

    def load_csqa(self):
        self.raw_csqa.clear()
        f = open(os.path.join(self.args.dataset_dir, 'csqa', f"{self.dataset_type}_rand_split.jsonl"), 'r', encoding='utf-8')
        for line in f:
            self.raw_csqa.append(json.loads(line.strip()))
        f.close()

    def make_dataloader(self, tokenizer, args, shuffle=True):
        batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size
        drop_last = False

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []

        for example in tqdm(self.examples):
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

    def set_predict_labels(self, predict_list):
        for index, case in enumerate(self.raw_csqa):
            case['AnswerKey_pred'] = chr(ord('A') + predict_list[index])
        return self.raw_csqa

    @staticmethod
    def load_example(case, cs4choice):
        pass
        # override to choose Example
        # return OMCSExample.load_from(case, cs4choice)

    def make_dev(self, predict_list, logits_list):
        # override
        for index, case in enumerate(self.raw_csqa):
            case['AnswerKey_pred'] = chr(ord('A') + predict_list[index])
            question = case['question']
            case.update(question)
            del case['question']
            for index_2, choice in enumerate(question['choices']):
                choice['logit'] = logits_list[index][index_2]

        return self.raw_csqa


class Baseline_Processor(ProcessorBase):
    
    def __init__(self, args, dataset_type):
        super(Baseline_Processor, self).__init__(args, dataset_type)

    def load_data(self):
        
        self.load_csqa()
        # convert raw data 2 CSQAexample
        self.examples = []
        for _, case in enumerate(self.raw_csqa):
            example = CSQAExample.load_from_json(case)
            self.examples.append(example)

        # self.examples = data

    def make_dataloader(self, tokenizer, args, shuffle=True):
        return super().make_dataloader(tokenizer, args, shuffle=shuffle)

    def make_dataloader_old(self, tokenizer, args, shuffle=True):
        batch_size = args.train_batch_size if self.dataset_type == 'train' else args.evltest_batch_size
        drop_last = False

        T, L = [], []

        for example in tqdm(self.examples):
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

        tensors.extend((all_input_ids, all_input_mask, all_segment_ids))
        
        # labels
        tensors.append(torch.tensor(self.data[1], dtype=torch.long))
        # b, 5, len; b,

        dataset = TensorDataset(*tensors)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        return dataloader


class OMCS_Processor(ProcessorBase):
    '''
    add multi cs at the end of the sequence.
    cs_num, max_seq_len
    '''
    
    def __init__(self, args, dataset_type):
        super(OMCS_Processor, self).__init__(args, dataset_type)
        self.omcs_version = args.OMCS_version

    def load_data(self):
        self.load_csqa()    # csqa dataset
        self.load_omcs()
        self.inject_commonsense()

    def load_omcs(self):
        dir_dict = {'1.0':'omcs_v1.0', '3.0':'omcs_v3.0_15', '3.1':'omcs_v3.1_10'}

        omcs_file = os.path.join(self.dataset_dir, 'omcs', dir_dict[self.omcs_version] ,f"{self.dataset_type}_rand_split_omcs.json")

        with open(omcs_file, 'r', encoding='utf-8') as f:
            self.omcs_cropus = json.load(f)
    
    @staticmethod
    def load_example(case, cs4choice):
        return OMCSExample.load_from(case, cs4choice)

    def inject_commonsense(self):
        omcs_index = 0
        for case in self.raw_csqa:
            cs4choice = {}
            for choice in case['question']['choices']:
                choice_text = choice['text']
                cs_list = self.omcs_cropus[omcs_index]['cs_list'][:self.args.cs_num]
                omcs_index += 1

                # cs_list.sort(key=lambda x:len(x)) # sort by cs_len

                temp = self.args.cs_num - len(cs_list)
                if temp:
                    cs_list.extend(['<unk>']*temp)

                # if len(cs_list) == 0:
                #     cs_list.append('choice_text')
                #     cs_list.append(question['question_concept'])
                    
                # distance = self.args.cs_num - len(cs_list)
                # while distance > 0:
                #     cs_list.extend(cs_list[:distance])
                #     distance = self.args.cs_num - len(cs_list)


                cs4choice[choice_text] = cs_list
            
            example = self.load_example(case, cs4choice)
            self.examples.append(example)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size
        drop_last = False

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []

        for example in tqdm(self.examples):
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
        

class Wiktionary_Processor(ProcessorBase):
    
    def __init__(self, args, dataset_type):
        super(Wiktionary_Processor, self).__init__(args, dataset_type)
        self.wkdt_version = args.WKDT_version

    def load_data(self):
        self.load_csqa()
        self.load_wkdt()
        self.inject_description()

    def load_wkdt(self):
        dir_dict = {'2.0': 'wiktionary_v2', '3.0': 'wiktionary_v3', '4.0': "wiktionary_v4", '5.0': "wiktionary_v5"}

        wiktionary_file = os.path.join(
            self.dataset_dir, 'wkdt', dir_dict[self.wkdt_version], 
            f"{self.dataset_type}_concept.json"
        )
        
        with open(wiktionary_file, 'r', encoding='utf-8') as f:
            self.wiktionary = json.load(f)
        

    def inject_description(self):
        if self.wkdt_version == "5.0":
            for key, value in self.wiktionary.items():
                self.wiktionary[key] = value[0]

        for case in self.raw_csqa:
            desc_dict = {}    # question concept, choice
            Qconcept = case['question']['question_concept']
            Qconcept_desc = self.wiktionary[Qconcept]
            desc_dict[Qconcept] = Qconcept_desc
            
            question = case['question']
            for choice in question['choices']:
                choice_text = choice['text']
                choice_desc = self.wiktionary[choice_text]
                desc_dict[choice_text] = choice_desc
                choice['desc'] = choice_desc
        
            case['Qconcept_desc'] = desc_dict[Qconcept]
            example = self.load_example(case, desc_dict)
            self.examples.append(example)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        return super(Wiktionary_Processor, self).make_dataloader(tokenizer, args, shuffle=shuffle)

    @staticmethod
    def load_example(case, cs4choice):
        return WKDTExample.load_from(case, cs4choice)

    def make_dev(self, predict_list, logits_list):
        super(Wiktionary_Processor, self).make_dev(predict_list, logits_list)
        return self.raw_csqa


class MSKE_Processor(OMCS_Processor, Wiktionary_Processor):
    '''
    Multi-Source Knowledge Ensemble
    '''

    def __init__(self, args, dataset_type):
        super(MSKE_Processor, self).__init__(args, dataset_type)
        self.dev_method = None
        self.ke_method_list = ["024", "135", "25", "34", "01", "top3"]
        # self.ke_method_list = ['shuffle3', 'shuffle3', 'shuffle2', 'shuffle2', 'top2', 'odd', 'even']

        if dataset_type in ['dev', 'test']:
            self.dev_method = args.dev_method
            logger.info(f"dev method {args.dev_method}")
        elif dataset_type == 'train':
            self.train_method = args.train_method
            logger.info(f"dev method {args.train_method}")

    def load_data(self):
        self.load_csqa()
        self.load_omcs()
        self.load_wkdt()
        if self.dataset_type not in ['dev', 'test'] or self.dev_method is not None:
            self.inject_wkdt_omcs()

    def remake_data(self, method):
        self.dev_method = method
        self.load_csqa()
        self.load_omcs()
        self.load_wkdt()
        self.inject_wkdt_omcs()

    def inject_wkdt_omcs(self):
        omcs_index = 0
        self.examples.clear()

        for case in self.raw_csqa:
            desc_dict = {}
            Qconcept = case['question']['question_concept']
            Qconcept_desc = self.wiktionary[Qconcept]
            desc_dict[Qconcept] = Qconcept_desc

            cs4choice = {}

            question = case['question']
            for choice in question['choices']:
                # 处理每一个 choice
                choice_text = choice['text']
                choice_desc = self.wiktionary[choice_text]
                desc_dict[choice_text] = choice_desc
                choice['desc'] = choice_desc

                cs_list = self.omcs_cropus[omcs_index]['cs_list'][:self.args.cs_num]
                omcs_index += 1
                cs4choice[choice_text] = cs_list

                temp = self.args.cs_num - len(cs_list)
                if temp:
                    cs_list.extend([' ']*temp)

                cs4choice[choice_text] = cs_list

            case['Qconcept_desc'] = desc_dict[Qconcept]

            method = self.dev_method if self.dataset_type in ['dev', 'test'] else self.train_method
            example = MSKEExample.load_from(case, cs4choice, desc_dict, method=method)
            self.examples.append(example)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        drop_last = False

        if self.dataset_type in ['dev', 'test'] and self.dev_method is None:
            return None

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []
        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []

        for example in tqdm(self.examples):
            feature_dict, labels = example.tokenize(tokenizer, args)
            all_input_ids.extend(feature_dict['input_ids'])
            all_token_type_ids.extend(feature_dict['token_type_ids'])
            all_attention_mask.extend(feature_dict['attention_mask'])
            all_label.extend(labels)

        all_input_ids = torch.stack(all_input_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        all_token_type_ids = torch.stack(all_token_type_ids)
        all_label = torch.tensor(all_label, dtype=torch.long)

        data = (all_input_ids, all_attention_mask, all_token_type_ids, all_label)

        dataset = TensorDataset(*data)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, drop_last=drop_last)

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
        return CSLinearExample.load_from(case, cs4choice)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        return super().make_dataloader(tokenizer, args, shuffle=shuffle)


class CSLinearEnhanced_Processor(OMCS_Processor):
    '''
    Base on OMCS_Processor, add restriction to the sequence len.
    question_len, cs_len, cs_num, max_qa_len 54, max_cs_len 18
    '''
    
    def __init__(self, args, dataset_type):
        super().__init__(args, dataset_type)
        self.max_qa_len = args.max_qa_len
        self.max_cs_len = args.max_cs_len
        self.max_seq_len = args.max_seq_len

        self.ke_method_list = ["024", "135", "25", "34", "01", "top3"]

        if dataset_type in ['dev', 'test']:
            self.dev_method = args.dev_method
            logger.info(f"dev method {args.dev_method}")
        elif dataset_type == 'train':
            self.train_method = args.train_method
            logger.info(f"dev method {args.train_method}")
    
    def load_data(self):
        self.load_csqa()
        self.load_omcs()
        if self.dataset_type not in ['dev', 'test'] or self.dev_method is not None:
            self.inject_omcs()

    def inject_omcs(self):
        omcs_index = 0
        self.examples.clear()

        for case in self.raw_csqa:
            cs4choice = {}
            question = case['question']
            for choice in question['choices']:
                # 处理每一个 choice
                choice_text = choice['text']
                cs_list = self.omcs_cropus[omcs_index]['cs_list'][:self.args.cs_num]
                omcs_index += 1
                cs4choice[choice_text] = cs_list

                if len(cs_list) == 0:
                    cs_list.append('choice_text')
                    cs_list.append(question['question_concept'])
                    
                distance = self.args.cs_num - len(cs_list)
                while distance > 0:
                    cs_list.extend(cs_list[:distance])
                    distance = self.args.cs_num - len(cs_list)

                cs4choice[choice_text] = cs_list

            method = self.dev_method if self.dataset_type in ['dev', 'test'] else self.train_method
            example = CSLinearEnhanceExample.load_from(case, cs4choice, method=method)
            self.examples.append(example)

    @staticmethod
    def load_example(case, cs4choice):
        return CSLinearExample.load_from(case, cs4choice)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        drop_last = False

        if self.dataset_type in ['dev', 'test'] and self.dev_method is None:
            return None

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []

        for example in tqdm(self.examples):
            feature_dict, labels = example.tokenize(tokenizer, args)
            all_input_ids.extend(feature_dict['input_ids'])
            all_token_type_ids.extend(feature_dict['token_type_ids'])
            all_attention_mask.extend(feature_dict['attention_mask'])
            all_label.extend(labels)

        all_input_ids = torch.stack(all_input_ids)
        all_attention_mask = torch.stack(all_attention_mask)
        all_token_type_ids = torch.stack(all_token_type_ids)
        all_label = torch.tensor(all_label, dtype=torch.long)

        data = (all_input_ids, all_attention_mask, all_token_type_ids, all_label)

        dataset = TensorDataset(*data)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, drop_last=drop_last)

        return dataloader


class OMCS_rerank_Processor(OMCS_Processor):

    def __init__(self, args, dataset_type):
        super().__init__(args, dataset_type)

    @staticmethod
    def load_example(case, cs4choice):
        return OMCSExample.load_from(case, cs4choice, mode='rerank')