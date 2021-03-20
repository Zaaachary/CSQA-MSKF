import json
from os import truncate
import pdb

import torch
from torch.utils.data import TensorDataset
from utils import feature
from utils.feature import Feature

class CSQAExample:
    '''
    "[CLS] question [SEP] question_concept [SEP] Choice [SEP]"
    '''

    def __init__(self, example_id, label, text_list):
        self.example_id = example_id
        self.text_list = text_list
        self.label = label

    def tokenize(self, tokenizer, max_seq_len):
        feature_list = []
        for text in self.text_list:
            tokens = tokenizer.tokenize(text)   # 分词 
            # 转换到 feature: (idx, input_ids, input_mask, segment_ids)
            feature = Feature.make_single(self.example_id, tokens, tokenizer, max_seq_len)
            feature_list.append(feature)
        # import pdb; pdb.set_trace()
        return feature_list, self.label
        
    @classmethod
    def load_from_json(cls, json_obj):
        '''
        通过 json 构造一个 CSQA Example
        '''
        example_id = json_obj['id']
        question = json_obj['question']['stem']
        question_concept = json_obj['question']['question_concept']
        choices = json_obj['question']['choices']
        label = ord(json_obj.get('answerKey', 'A')) - ord('A')
        
        text_list = ['' for _ in range(5)]
        for index, choice in enumerate(choices):
            text_list[index] = f" {question} [SEP] {question_concept} [SEP] {choice['text']} "

        return cls(example_id, label, text_list)


class OMCSExample(object):
    '''
    "[CLS] question [SEP] question_concept [SEP] Choice [SEP] cs_1 [SEP] ... [SEP] cs_n [SEP]"
    '''

    def __init__(self, example_id, label, text_list):
        self.example_id = example_id
        self.label = label
        self.text_list = text_list
    
    def __repr__(self) -> str:
        return f'{self.example_id}: {self.label}'

    def tokenize(self, tokenizer, max_seq_len):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        '''
        feature_dict = tokenizer.batch_encode_plus(self.text_list, add_special_tokens=False, max_length=max_seq_len, padding='max_length', truncation =True, return_tensors='pt')
        # import pdb; pdb.set_trace()
        return feature_dict

    @staticmethod
    def make_text(question, choices, cs4choice, question_concept):
        """
        organize the content_dict to text_list; rewrite !
        "[CLS] question [SEP] question_concept [SEP] Choice [SEP] cs_1 [SEP] ... [SEP] cs_n [SEP]"
        """
        text_list = []
        for choice in choices:
            # choice: {'label':xx , 'text':xx}
            choice_str = choice['text']
            text = f"[CLS] {question} [SEP] {question_concept} [SEP] {choice_str} [SEP]"
            for cs in cs4choice[choice_str]:
                text += f" {cs} [SEP]"
            text_list.append(text)
        return text_list

    @classmethod
    def load_from(cls, case, cs4choice):

        example_id = case['id']
        label = ord(case.get('answerKey', 'A')) - ord('A')
        question = case['question']['stem']
        question_concept = case['question']['question_concept']
        choices = case['question']['choices']
        
        text_list = cls.make_text(question, choices, cs4choice, question_concept)
        return cls(example_id, label, text_list)
        

class CSLinearExample(OMCSExample):

    def __init__(self, example_id, label, text_list):
        super().__init__(example_id, label, text_list)

    def tokenize(self, tokenizer, max_len_tuple):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        [CLS] question [SEP] question_concept [SEP] Choice [SEP] PADDING cs_1 [SEP] ... [SEP] cs_n [SEP]
        '''
        all_feature_dict = {}
        max_qa_len, max_cs_len = max_len_tuple

        all_feature_list = []   # [qc_featre cat cs_feature,  ...]
        for case in self.text_list:
            qa_list, cs_list = case

            qa_feature_dict = tokenizer.encode_plus(qa_list[0], qa_list[1], add_special_tokens=True, max_length=max_qa_len, padding='max_length', truncation='only_first', return_tensors='pt')
            # import pdb; pdb.set_trace()

            cs_total_feature_dict = {}
            # cs_total_feature_dict = {'input_ids':, 'token_type_ids', 'attention_mask'}
            for cs in cs_list:
                cs_feature_dict = tokenizer.encode_plus(cs, add_special_tokens=False, max_length=max_cs_len, padding='max_length', truncation=True, return_tensors='pt')

                cs_total_feature_dict = self.concat_feature_dict(cs_total_feature_dict, cs_feature_dict)

            all_feature_list.append(self.concat_feature_dict(qa_feature_dict, cs_total_feature_dict))

        keys = ('input_ids', 'token_type_ids', 'attention_mask')
        for key in keys:
            target_list = [case[key] for case in all_feature_list]
            # print([case.shape for case in target_list])
            # import pdb; pdb.set_trace()
            all_feature_dict[key] = torch.stack(target_list, dim=0)
            all_feature_dict[key] = torch.squeeze(all_feature_dict[key], dim=1)
        
        return all_feature_dict

    @staticmethod
    def concat_feature_dict(feature_dict1, feature_dict2):
        if len(feature_dict1) == 0:
            return feature_dict2

        keys = ('input_ids', 'token_type_ids', 'attention_mask')
        for key in keys:
            # feature_dict1[key] [1, seq_len]
            temp_tensor = torch.cat((feature_dict1[key], feature_dict2[key]), dim=1)
            feature_dict1[key] = temp_tensor

        return feature_dict1
            

    @staticmethod
    def make_text(question, choices, cs4choice, question_concept):
        """
        "[CLS] question [SEP] question_concept [SEP] Choice [SEP] cs_1 [SEP] ... [SEP] cs_n [SEP]"

        return text_list: 
        [# all qa pair
            [ # case for a qa pair
                ["question [SEP]", "question_concept [SEP] Choice"], 
                ["cs_n [SEP]", ...]
            ],
            ...
        ]
        """
        text_list = []
        for choice in choices:
            choice_str = choice['text']
            qa_list = [f"{question} [SEP]", f"{question_concept} [SEP] {choice_str}"]
            cs_list = [f"{cs} [SEP]" for cs in cs4choice[choice_str]]            
            text_list.append((qa_list, cs_list))
        return text_list