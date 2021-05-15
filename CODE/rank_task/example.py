import json
import random

import torch

from csqa_task.example import BaseExample

class OMCSrankExample(BaseExample):

    def __init__(self, example_id, label, text_list):
        
        super(OMCSrankExample, self).__init__(example_id, label)
        self.text_list = text_list

    def tokenize(self, tokenizer, args):
        max_seq_len = args.max_seq_len
        feature_dict = tokenizer.batch_encode_plus(self.text_list, add_special_tokens=True, max_length=max_seq_len, padding='max_length', truncation='only_first', return_tensors='pt')

        labels = [self.label] * len(self.text_list)
        return feature_dict, labels

    @staticmethod
    def make_text(question, question_concept, choice, cs_list):
        
        text_list = []

        for cs in cs_list:

            text = f"{choice} [SEP] {question}", f"{cs[1]}"

            text_list.append(text)

        return text_list

    @classmethod
    def load_from(cls, case, cs_list, isgood):
        
        label = 1 if isgood else 0

        example_id = case['id']
        question = case['question']
        question_concept = case['question_concept']
        choice = case['choice']
        
        text_list = cls.make_text(question, question_concept, choice, cs_list)
        return cls(example_id, label, text_list)