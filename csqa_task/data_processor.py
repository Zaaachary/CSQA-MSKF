#! -*- encoding:utf-8 -*-
"""
@File    :   data_processor.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import json

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from csqa_task.example import CSQAExample

class MSBaseline_Processor():
    
    def __init__(self, data_dir, dataset_type):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.data = None

    def load_data(self):
        data = []

        # load raw data
        f = open(os.path.join(self.data_dir, 'csqa', f"{self.dataset_type}_rand_split.json"), 'r', encoding='utf-8')
        for line in f:
            data.append(line.strip())
        f.close()

        # convert raw data 2 CSQAexample
        for index, case in enumerate(data):
            case = json.loads(case)
            example = CSQAExample.load_from_json(case)
            data[index] = example

        self.data = data

    def make_dataloader(self, tokenizer, batch_size, drop_last, max_seq_len, shuffle=True):
        T, L = [], []

        for example in tqdm(self.data):
            text_list, label = example.tokenize(tokenizer, max_seq_len)
            
            T.append((text_list))
            L.append(label)
        
        self.data = (T, L)  # len(T) = len(L)
        return self._convert_to_tensor(batch_size, drop_last, shuffle)

    def _convert_to_tensor(self, batch_size, drop_last, shuffle):
        tensors = []

        features = self.data[0]     # tensor, label
        all_idx = torch.tensor([[f.idx for f in fs] for fs in features], dtype=torch.long)
        all_input_ids = torch.tensor([[f.input_ids for f in fs] for fs in features], dtype=torch.long)
        all_input_mask = torch.tensor([[f.input_mask for f in fs] for fs in features], dtype=torch.long)
        all_segment_ids = torch.tensor([[f.segment_ids for f in fs] for fs in features], dtype=torch.long)

        # features
        # tensors.extend((all_input_ids, all_input_mask, all_segment_ids))
        tensors.extend((all_idx, all_input_ids, all_input_mask, all_segment_ids))
        
        # labels
        tensors.append(torch.tensor(self.data[1], dtype=torch.long))
        # b, 5, len; b,

        dataset = TensorDataset(*tensors)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        return dataloader

if __name__ == "__main__":
    a = CSQAProcessor('DATA\\', 'dev')
    a.load_data()
