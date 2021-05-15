import os
import json
import pdb
from random import random, sample
from copy import deepcopy
import logging

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from csqa_task.data import ProcessorBase
from csqa_task.example import OMCSExample


class RankOMCS_Processor(ProcessorBase):
    '''
    add multi cs at the end of the sequence.
    cs_num, max_seq_len
    '''
    
    def __init__(self, args, dataset_type):
        super(RankOMCS_Processor, self).__init__(args, dataset_type)
        self.omcs_version = args.OMCS_version
        self.csqa_cs_list = []

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
        for case_index, case in enumerate(self.raw_csqa[:2]):
            question = case['question']
            # 为每个 target choice 评估它的每条常识
            for choice_index, target_choice in enumerate(question['choices']):
                target_choice_text = target_choice['text']
                target_cs_index = case_index * 5 + choice_index
                target_cs_list = self.omcs_cropus[target_cs_index]['cs_list'][:self.args.cs_num]

                target_choice_info = {
                    'id': case['id'],
                    'question': question['stem'],
                    'question_concept': question['question_concept'],
                    'choice': target_choice_text,
                    'isanswer': target_choice['label'] == case['answerKey'],
                    'cs_list': target_cs_list
                }

                for cs in target_cs_list:
                    omcs_index = case_index * 5
                    cs4choice = {}

                    for choice in question['choices']:
                        choice_text = choice['text']

                        if choice_text == target_choice_text:
                            insert_cs = cs
                        else:
                            cs_list = self.omcs_cropus[omcs_index]['cs_list'][:self.args.cs_num]
                            if len(cs_list) == 0:
                                cs_list = ['<unk>',]
                            insert_cs = cs_list[-1]

                        omcs_index += 1
                        choice['cs'] = insert_cs
                        cs4choice[choice_text] = [insert_cs, ]

                    example = self.load_example(case, cs4choice)
                    self.examples.append(example)

                self.csqa_cs_list.append(target_choice_info)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size
        drop_last = False

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_label = []
        # import pdb; pdb.set_trace()
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

    def set_cs_loss(self, loss_list):
        # loss_list  cs_num * 5 * B
        loss_index = 0
        for case in self.csqa_cs_list:
            for cs_index, cs in enumerate(case['cs_list']):
                case['cs_list'][cs_index] = (loss_list[loss_index], cs)
                loss_index += 1

            if case['isanswer']:
                # 正确答案 则 loss 从小到大排序
                case['cs_list'].sort(key=lambda x: x[0])
            else:
                # 错误答案 则 loss 从大到小排序
                case['cs_list'].sort(key=lambda x: x[0], reverse=True)

        return self.csqa_cs_list