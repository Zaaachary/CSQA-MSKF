import json

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
        
        return feature_list, self.label
        
    @classmethod
    def load_from_json(cls, json_obj):
        '''
        通过 json 构造一个 CSQA Example
        '''
        example_id = json_obj['idx']
        question = json_obj['question']['stem']
        question_concept = json_obj['question']['question_concept']
        choices = json_obj['question']['choices']
        label = ord(json_obj.get('answerKey', 'A')) - ord('A')
        
        text_list = ['' for _ in range(5)]
        for index, choice in enumerate(choices):
            text_list[index] = f" {question} [SEP] {question_concept} [SEP] {choice['text']} "

        return cls(example_id, label, text_list)


class OMCSExample(object):

    def __init__(self, example_id, label, content_dict):
        self.example_id = example_id
        self.label = label
        self.content_dict = content_dict
        self.text_list = []
    
    def make_text(self):
        """
        organize the content_dict to text_list; rewrite !
        """
        pass

    @classmethod
    def load_from_json(cls, json_obj):
        """
        通过 json 构造一个 OMCS Example
        """
        pass