#! -*- encoding:utf-8 -*-
"""
@File    :   AlbertBurger.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import math
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AlbertModel as tfms_AlbertModel
from transformers import AlbertPreTrainedModel, AlbertConfig
from .AlbertModel import AlbertModel

from utils import common

class CSLinearBase(object):

    def __init__(self) -> None:
        self.cs_range_list = None
        self.qa_range_list = None

    def _pad_qacs_to_maxlen(self, flat_input_ids, last_hidden_state):
        '''
        input
        - last_hidden_state [5B, seq_len, hidden]

        return
        - cs_range_list: [B*5, cs_num]  (start, end)  sep+1, sep
        - qa_range_list: [B*5]  (end)
        - cs_encoding: [B*5, cs_num, max_cs_len, H]
        - qa_encoding: [B*5, cs_num, max_qa_len, H]
        - cs_attn_mask
        - qa_attn_mask
        '''
        # Locate SEP token
        input_ids = flat_input_ids.cpu().clone().detach().numpy()
        sep_ids = input_ids == 3    # sep toekn in albert is 3
        sep_locate = [[] for _ in range(len(sep_ids))]  # [B*5, seq_num]
        for index_1, case in enumerate(sep_ids):
            for index_2, token in enumerate(case):
                if token:
                    sep_locate[index_1].append(index_2)
        # Get CS, QA range
        self.cs_range_list = [[] for _ in range(len(sep_ids))]   # [B*5, cs_num]
        self.qa_range_list = []
        for index, case in enumerate(sep_locate):
            # Q [S] QC [S] Choice [S] cs_1[S] cs_2[S]    
            # qa: Q [S] QC [S] Choice [S]; cs: cs_1[S]
            self.qa_range_list.append(case[2]+1)
            start = case[2]
            for end in case[3:]:
                cs_tuple = (start+1, end+1)
                start = end
                self.cs_range_list[index].append(cs_tuple)

        # Get CS and stack to tensor
        hidden_size = last_hidden_state.shape[-1]
        cs_batch_list, cs_padding_batch_list = [],[]
        for index, case in enumerate(self.cs_range_list):
            cs_case_list = []
            cs_padding_list = []
            for cs in case:
                start, end = cs
                pad_len = self.max_cs_len - (end-start)

                cs = last_hidden_state[index, start:end, :]
                zero = torch.zeros(pad_len, hidden_size, dtype=last_hidden_state.dtype)
                zero = zero.to(last_hidden_state.device)
                cs_case_list.append(torch.cat((cs, zero), dim=-2))

                mask = torch.cat((torch.zeros(cs.shape[:-1]), torch.ones(pad_len))).bool()
                mask = mask.to(last_hidden_state.device)
                cs_padding_list.append(mask)

            cs_batch_list.append(torch.stack(cs_case_list))
            cs_padding_batch_list.append(torch.stack(cs_padding_list))

        cs_encoding = torch.stack(cs_batch_list)
        cs_padding_mask = torch.stack(cs_padding_batch_list)

        # Get QA and stack to tensor
        qa_batch_list, qa_padding_batch_list = [], []
        for index, case in enumerate(self.qa_range_list):
            end = case
            pad_len = self.max_qa_len - (end-1)

            qa = last_hidden_state[index, 1:end, :]  # [CLS] -> [SEP]  doesn't contain CLS
            zero = torch.zeros(pad_len, hidden_size, dtype=last_hidden_state.dtype)
            zero = zero.to(last_hidden_state.device)
            qa_batch_list.append(torch.cat((qa, zero), dim=-2))

            mask = torch.cat((torch.zeros(qa.shape[:-1]), torch.ones(pad_len))).bool()
            mask = mask.to(last_hidden_state.device)
            qa_padding_batch_list.append(mask)

        qa_encoding = torch.stack(qa_batch_list)
        # qa_encoding = qa_encoding.unsqueeze(1).expand(-1, self.cs_num, -1, -1)
        qa_padding_mask = torch.stack(qa_padding_batch_list)
        # qa_padding_mask = qa_padding_mask.unsqueeze(1).expand(-1, self.cs_num, -1)

        return cs_encoding, cs_padding_mask, qa_encoding, qa_padding_mask

    def _remvoe_cs_pad_add_to_last_hidden_state(self, cs_encoding, last_hidden_state):
        self.cs_range_list    # [[(start, end), (start, end)], [], [],]
        self.qa_range_list    # [end, end, end,]
        for index, cs_range in enumerate(self.cs_range_list):
            for cs_index, cs_case in enumerate(cs_range):
                start, end = cs_case
                last_hidden_state[index, start:end] = cs_encoding[index, cs_index,:end-start,:]
    
        return last_hidden_state