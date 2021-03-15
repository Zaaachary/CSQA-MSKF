import json
from os import truncate
import pdb

from torch.utils.data import TensorDataset
from utils.feature import Feature

class CSQAExample:
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

        return feature_dict

    @staticmethod
    def make_text(question, choices, cs4choice, question_concept):
        """
        organize the content_dict to text_list; rewrite !
        "[CLS] question [SEP] Choice [SEP] cs_1 [SEP] ... [SEP] cs_n [SEP]"
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
        