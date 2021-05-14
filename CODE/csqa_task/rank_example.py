import json
import random

import torch

from csqa_task.example import BaseExample


class RankOMCSExample(BaseExample):

    def __init__(self, example_id, label, text_list):
        super(RankOMCSExample, self).__init__(example_id, label)
        self.text_list = text_list

    def tokenize(self, tokenizer, args):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        '''
        max_seq_len = args.max_seq_len
        feature_dict = tokenizer.batch_encode_plus(self.text_list, add_special_tokens=False, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
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