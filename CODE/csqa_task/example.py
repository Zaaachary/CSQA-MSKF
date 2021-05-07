import json
from os import truncate
import pdb
import random

import torch
from utils.feature import Feature


class BaseExample(object):

    def __init__(self, example_id, label) -> None:
        self.example_id = example_id
        self.label = label

    def __repr__(self) -> str:
        return f'{self.example_id}: {self.label}'

    def tokenize(self, tokenizer, args):
        '''
        override
        '''
        max_seq_len = args.max_seq_len
        feature_dict = tokenizer.batch_encode_plus(self.text_list, add_special_tokens=False, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
        pass
    
    @classmethod
    def load_from(cls):
        pass


class CSQAExample:
    '''
    "[CLS] question [SEP] question_concept [SEP] Choice [SEP]"
    '''

    def __init__(self, example_id, label, text_list):
        self.example_id = example_id
        self.text_list = text_list
        self.label = label

    def tokenize_old(self, tokenizer, args):
        max_seq_len = args.max_seq_len
        feature_list = []
        for question, choice in self.text_list:
            text = question + "[SEP]" + choice
            tokens = tokenizer.tokenize(text)   # 分词 
            # 转换到 feature: (idx, input_ids, input_mask, segment_ids)
            feature = Feature.make_single(self.example_id, tokens, tokenizer, max_seq_len)
            feature_list.append(feature)

        return feature_list, self.label

    def tokenize(self, tokenizer, args):
        max_seq_len = args.max_seq_len

        feature_dict = tokenizer.batch_encode_plus(self.text_list, add_special_tokens=True, max_length=max_seq_len, padding='max_length', truncation='only_first', return_tensors='pt')

        return feature_dict
            
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
            # text_list[index] = f" {question} [SEP] {question_concept} [SEP] {choice['text']} "
            text_list[index] = f" {question} ", f" {question_concept} [SEP] {choice['text']} "

        return cls(example_id, label, text_list)


class OMCSExample(BaseExample):
    '''
    origin mode
    "[CLS] question [SEP] question_concept [SEP] Choice [SEP] cs_1 [SEP] ... [SEP] cs_n [SEP]"

    rerank mode
    ""
    '''

    def __init__(self, example_id, label, text_list, mode):
        super(OMCSExample, self).__init__(example_id, label)
        self.text_list = text_list
        self.mode = mode

    def tokenize(self, tokenizer, args):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        '''
        max_seq_len = args.max_seq_len
        feature_dict = tokenizer.batch_encode_plus(self.text_list, add_special_tokens=False, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
        return feature_dict

    @staticmethod
    def make_text(question, choices, cs4choice, question_concept, mode):
        """
        organize the content_dict to text_list; rewrite !
        "[CLS] question [SEP] question_concept [SEP] Choice [SEP] cs_1 [SEP] ... [SEP] cs_n [SEP]"
        """
        text_list = []
        for choice in choices:
            # choice: {'label':xx , 'text':xx}
            choice_str = choice['text']
            text = f"[CLS] {question} [SEP] {question_concept} [SEP] {choice_str} [SEP]"

            if mode == "origin":
                for cs in cs4choice[choice_str]:
                    text += f" {cs} [SEP]"
                text_list.append(text)

            elif mode == "rerank":
                cs_text = []
                for cs in cs4choice[choice_str]:
                    cs_text.append(text + f" {cs} [SEP]")
                text_list.extend(cs_text)

        return text_list

    @classmethod
    def load_from(cls, case, cs4choice, mode='origin'):

        example_id = case['id']
        label = ord(case.get('answerKey', 'A')) - ord('A')
        question = case['question']['stem']
        question_concept = case['question']['question_concept']
        choices = case['question']['choices']
        
        text_list = cls.make_text(question, choices, cs4choice, question_concept, mode)
        return cls(example_id, label, text_list, mode)
        

class CSLinearExample(OMCSExample):

    def __init__(self, example_id, label, text_list):
        super().__init__(example_id, label, text_list)

    def tokenize(self, tokenizer, args):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        [CLS] question [SEP] question_concept [SEP] Choice [SEP] PADDING cs_1 [SEP] ... [SEP] cs_n [SEP]
        '''
        max_qa_len = args.max_qa_len
        max_cs_len = args.max_cs_len
        max_seq_len = args.max_seq_len
        self.cut_add(tokenizer, max_qa_len, max_cs_len)

        all_qa_ids = [case[0] for case in self.text_list]
        all_cs_ids = [case[1] for case in self.text_list]

        feature_dict = tokenizer.batch_encode_plus(list(zip(all_qa_ids, all_cs_ids)), add_special_tokens=True, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')

        return feature_dict

    def cut_add(self, tokenizer, max_qa_len, max_cs_len):
        sep = tokenizer.sep_token
        max_qa_len -= 2  # current qa doesn't contain cls and endsep

        for index, case in enumerate(self.text_list):
            qa_list, cs_list = case
            qa_ids = tokenizer.tokenize(f' {sep} '.join(qa_list))
            if len(qa_ids) > max_qa_len:
                qa_ids = qa_ids[len(qa_ids)-max_qa_len:]
            
            cs_ids = []
            for j, cs in enumerate(cs_list):
                temp = tokenizer.tokenize(cs)
                temp = temp[:max_cs_len-1] # last place for sep
                if j != len(cs_list)-1:
                    temp = temp + [tokenizer.sep_token]
                cs_ids.extend(temp)

            self.text_list[index] = qa_ids, cs_ids

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
            qa_list = [question, question_concept, choice_str]
            cs_list = [cs for cs in cs4choice[choice_str]]
            text_list.append((qa_list, cs_list))
        return text_list


class WKDTExample(BaseExample):
    '''
    [CLS] Question Chocie [SEP] Q_Concept description [SEP] Chocie description [SEP]
    '''

    def __init__(self, example_id, label, text_list):
        super().__init__(example_id, label)
        self.text_list = text_list
    
    def tokenize(self, tokenizer, args):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        '''
        max_qa_len = args.max_qa_len
        max_desc_len = args.max_desc_len
        max_seq_len = args.max_seq_len
        self.cut_add(tokenizer, max_qa_len, max_desc_len)

        all_qa_ids = [case[0] for case in self.text_list]
        all_desc_ids = [case[1] for case in self.text_list]
        
        # import pdb; pdb.set_trace()
        feature_dict = tokenizer.batch_encode_plus(list(zip(all_qa_ids, all_desc_ids)), add_special_tokens=False, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')

        return feature_dict

    def cut_add(self, tokenizer, max_qa_len, max_desc_len):
        sep = tokenizer.sep_token
        cls = tokenizer.cls_token
        max_qa_len -= 2  # current qa doesn't contain cls and endsep
        max_desc_len -= 1

        for index, case in enumerate(self.text_list):
            case = [text.replace('[SEP]', sep) for text in case] 
            qa_text, qc_desc, c_desc = case
            qa_text = cls + qa_text

            qa_ids = tokenizer.tokenize(qa_text)
            if len(qa_ids) > max_qa_len:
                qa_ids = qa_ids[len(qa_ids)-max_qa_len:]

            qc_desc_ids = tokenizer.tokenize(qc_desc)
            if len(qc_desc_ids) > max_desc_len:
                qc_desc_ids = qc_desc_ids[len(qc_desc_ids)-max_desc_len:]

            c_desc_ids = tokenizer.tokenize(c_desc)
            if len(c_desc_ids) > max_desc_len:
                c_desc_ids = c_desc_ids[len(c_desc_ids)-max_desc_len:]

            self.text_list[index] = qa_ids, qc_desc_ids + c_desc_ids

    @staticmethod
    def make_text(question, choices, desc_dict, question_concept):
        Qconcept_desc = desc_dict[question_concept]
        text_list = []
        for choice in choices:
            choice_text = choice['text']
            choics_desc = desc_dict[choice_text]
            texts = [
                f" {question} {choice_text} [SEP]",  
                f" {question_concept}: {Qconcept_desc} [SEP]", 
                f" {choice_text}: {choics_desc} [SEP]"
            ]
            text_list.append(texts)
        return text_list

    @classmethod
    def load_from(cls, case, desc_dict):
        example_id = case['id']
        label = ord(case.get('answerKey', 'A')) - ord('A')
        question = case['question']['stem']
        question_concept = case['question']['question_concept']
        choices = case['question']['choices']

        text_list = cls.make_text(question, choices, desc_dict, question_concept)
        return cls(example_id, label, text_list)


class CSLinearExampleV1(OMCSExample):

    def __init__(self, example_id, label, text_list):
        super().__init__(example_id, label, text_list)

    def tokenize(self, tokenizer, args):
        '''
        feature_dict: 'input_ids', 'token_type_ids', 'attention_mask'
        [CLS] question [SEP] question_concept [SEP] Choice [SEP] PADDING cs_1 [SEP] ... [SEP] cs_n [SEP]
        '''
        all_feature_dict = {}
        max_qa_len, max_cs_len = args.max_qa_len, args.max_cs_len

        all_feature_list = []   # [qc_featre cat cs_feature,  ...]
        for case in self.text_list:
            qa_list, cs_list = case

            qa_feature_dict = tokenizer.encode_plus(qa_list[0], qa_list[1], add_special_tokens=True, max_length=max_qa_len, truncation='only_first', return_tensors='pt')

            cs_total_feature_dict = {}
            # cs_total_feature_dict = {'input_ids':, 'token_type_ids', 'attention_mask'}
            for cs in cs_list:
                cs_feature_dict = tokenizer.encode_plus(cs, add_special_tokens=False, max_length=max_cs_len, truncation=True, return_tensors='pt')

                cs_total_feature_dict = self.concat_feature_dict(cs_total_feature_dict, cs_feature_dict)

            all_feature_list.append(self.concat_feature_dict(qa_feature_dict, cs_total_feature_dict))

        keys = ('input_ids', 'token_type_ids', 'attention_mask')
        for key in keys:
            target_list = [case[key] for case in all_feature_list]
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
            qa_list = [f"{question}", f"{question_concept} [SEP] {choice_str}"]
            cs_list = [f"{cs} [SEP]" for cs in cs4choice[choice_str]]            
            text_list.append((qa_list, cs_list))
        return text_list


class MSKEExample(BaseExample):

    def __init__(self, example_id, label, text_stack) -> None:
        super().__init__(example_id, label)
        # [[], [], [], ...]  double/tripe example with different cs
        self.text_stack = text_stack

    def tokenize(self, tokenizer, args):
        '''
        all_feature_dict = {
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': []
        }
        '''
        max_seq_len = args.max_seq_len
        # self.text_stack
        all_feature_dict = {
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': []
        }
        labels = [self.label for _ in range(len(self.text_stack))]
        for text_list in self.text_stack:
            # text_list [[qa, cs], [qa, cs], x5]
            feature_dict = tokenizer.batch_encode_plus(text_list, add_special_tokens=True, max_length=max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
            all_feature_dict['input_ids'].append(feature_dict['input_ids'])
            all_feature_dict['token_type_ids'].append(feature_dict['token_type_ids'])
            all_feature_dict['attention_mask'].append(feature_dict['attention_mask'])
        
        return all_feature_dict, labels

    @staticmethod
    def make_text_stack(question, question_concept, choices, desc_dict, cs4choice, method):

        cs_type_list = [
            'Qconcept_desc', 'Choice_desc', 'both_desc',
            'odd', 'even', 'origin', 'top2', 'shuffle3', 'shuffle2',
            ]

        def choose_cs_type(method):
            if method == 'trian_01':
                cs_type = ['odd', 'even']
                m1 = 'top2'
                m2 = random.choice(cs_type)
                return (m1, m2)
            elif method == 'train_01_equal':
                cs_type = ['odd', 'even', 'top2']
                m1 = random.choice(cs_type)
                cs_type.remove(m1)
                m2 = random.choice(cs_type)
                return (m1, m2)
            elif method == 'trian_02':
                cs_type = ['shuffle2', 'shuffle3']
                m1 = 'top2'
                m2 = random.choice(cs_type)
                return (m1, m2)
            elif method in cs_type_list:
                return (method, )

        if method in ['trian_01', 'trian_02', 'train_01_equal']:
            text_stack = [[], []]
        elif method in cs_type_list:
            text_stack = [[],]

        cstype_stack = []

        Qconcept_desc = desc_dict[question_concept]
        for choice in choices:
            choice_text = choice['text']
            choics_desc = desc_dict[choice_text]

            cs = {
                'odd': cs4choice[choice_text][1::2],
                'even': cs4choice[choice_text][::2],
                'top2': cs4choice[choice_text][:2],
                'shuffle2': random.sample(cs4choice[choice_text], k=2),
                'shuffle3': random.sample(cs4choice[choice_text], k=3),
            }

            text = f" {question} {choice_text} [SEP] {question_concept} [SEP] {choice_text} [SEP] "

            cstype_list = choose_cs_type(method)
            # import pdb; pdb.set_trace()
            cstype_stack.append(cstype_list)

            for index, cs_type in enumerate(cstype_list):
                if cs_type == 'Qconcept_desc':
                    text_temp = f" {question} {choice_text} [SEP] {question_concept} : {Qconcept_desc} [SEP] {choice_text} [SEP] "
                elif cs_type == 'Choice_desc':
                    text_temp = f" {question} {choice_text} [SEP] {question_concept} [SEP] {choice_text} : {choics_desc} [SEP] "
                elif cs_type == 'both_desc':
                    text_temp = f" {question} {choice_text} [SEP] {question_concept} : {Qconcept_desc} [SEP] {choice_text} : {choics_desc} [SEP] "

                elif cs_type in ['odd', 'even', 'top2', 'shuffle3', 'shuffle2']:
                    text_temp = text + ' [SEP] '.join(cs[cs_type])
                elif cs_type == 'origin':
                    text_temp = f"{question} {choice_text} [SEP] {question_concept} [SEP] {choice_text}"

                text_stack[index].append(text_temp)

        return text_stack

    @classmethod
    def load_from(cls, case, cs4choice, desc_dict, method):
        example_id = case['id']

        label = ord(case.get('answerKey', 'A')) - ord('A')
        question = case['question']['stem']
        question_concept = case['question']['question_concept']
        choices = case['question']['choices']

        text_stack = cls.make_text_stack(question, question_concept, choices, desc_dict, cs4choice, method)

        return cls(example_id, label, text_stack)