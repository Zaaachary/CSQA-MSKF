#! -*- encoding:utf-8 -*-
"""
@File    :   HH_linear.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AlbertModel, AlbertPreTrainedModel, BertModel,
                          BertPreTrainedModel)

from utils import common


class AlbertCrossAttn(AlbertPreTrainedModel):
    '''
    input_ids [b, 5, seq_len] => [5b, seq_len]
    => PTM
    cs_encoding [5b, cs_len, cs_seq_len, hidden]
    query_encoding [5b, query_len, hidden] => [5b, cs_len, query_len, hidden]
    => cross_attn
    qc_attoutput  [5b, cs_len, query_seq_len, hidden]
    cq_attoutput  [5b, cs_len, cs_seq_len, hidden]
    
    '''
    def __init__(self, config, **kwargs):
        super(AlbertCrossAttn, self).__init__(config)
        # length config
        self.cs_num = kwargs['cs_num']
        self.max_cs_len = kwargs['max_cs_len']
        self.max_qa_len  = kwargs['max_qa_len']

        # modules
        self.albert = AlbertModel(config)
        self.cross_att = AttentionLayer(config.hidden_size, self.cs_num)

        self.cs_merge = AttentionMerge(config.hidden_size, config.hidden_size//4)
        self.qu_merge = AttentionMerge(config.hidden_size, config.hidden_size//4)
        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 3, 1)
        )

        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)    
        return loss, right_num

    def _forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # [B, 5, L] => [B * 5, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.albert(
            input_ids = flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )

        pooler_output = outputs.pooler_output   # outputs[1]  CLS token    [5b, hidden]
        last_hidden_state = outputs.last_hidden_state   # outputs[0]     [5b, seq_len, hidden] 
        # separate query and commonsense encoding        
        # [C] Q [S] QC [S] C [S] cs_1 [S] ←cs_seq_len cs2 ...[S]

        cs_encoding, cs_padding_mask, qa_encoding, qa_padding_mask = self._pad_qacs_to_maxlen(flat_input_ids, last_hidden_state)
        # import pdb; pdb.set_trace()

        # cross-attn
        # [5b, cs_len, query_seq_len, H]
        qc_attn_output, qc_attn_weights = self.cross_att(qa_encoding, cs_encoding, cs_padding_mask)
        # [5b, cs_len, cs_seq_len, H]
        cq_attn_output, cq_attn_weights = self.cross_att(cs_encoding, qa_encoding, qa_padding_mask)

        # [5b, cs_len, cs_seq_len, hidden] => [5b, cs_seq_len, hidden]
        # [5b, cs_seq_len, hidden] => [5b, hidden]
        cs_rep = self.cs_merge(cq_attn_output)
        # cs_rep = torch.mean(cq_attoutput,dim = -3)
        cs_rep = torch.mean(cs_rep, dim = -2)

        # mean pooling query encoding
        qu_rep = self.qu_merge(qc_attn_output)
        # qu_rep = torch.mean(qc_attoutput, dim = -3)
        qu_rep = torch.mean(qu_rep, dim = -2)

        final_rep = torch.cat((pooler_output,cs_rep,qu_rep),dim = -1)
        logits = self.scorer(final_rep).view(-1, 5)

        return logits

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
        cs_range_list = [[] for _ in range(len(sep_ids))]   # [B*5, cs_num]
        qa_range_list = []
        for index, case in enumerate(sep_locate):
            # Q [S] QC [S] Choice [S] cs_1[S] cs_2[S]    
            # qa: Q [S] QC [S] Choice [S]; cs: cs_1[S]
            qa_range_list.append(case[2]+1)
            start = case[2]
            for end in case[3:]:
                cs_tuple = (start+1, end+1)
                start = end
                cs_range_list[index].append(cs_tuple)

        # Get CS and stack to tensor
        hidden_size = last_hidden_state.shape[-1]
        cs_batch_list, cs_padding_batch_list = [],[]
        for index, case in enumerate(cs_range_list):
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
        for index, case in enumerate(qa_range_list):
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
        qa_encoding = qa_encoding.unsqueeze(1).expand(-1, self.cs_num, -1, -1)
        qa_padding_mask = torch.stack(qa_padding_batch_list)
        qa_padding_mask = qa_padding_mask.unsqueeze(1).expand(-1, self.cs_num, -1)

        return cs_encoding, cs_padding_mask, qa_encoding, qa_padding_mask


class BertCrossAttn(BertPreTrainedModel):
    '''
    input_ids [b, 5, seq_len] => [5b, seq_len]
    => PTM
    cs_encoding [5b, cs_len, cs_seq_len, hidden]
    query_encoding [5b, query_len, hidden] => [5b, cs_len, query_len, hidden]
    => cross_attn
    qc_attoutput  [5b, cs_len, query_seq_len, hidden]
    cq_attoutput  [5b, cs_len, cs_seq_len, hidden]
    
    '''
    def __init__(self, config, **kwargs):
        super(BertCrossAttn, self).__init__(config)
        # length config
        self.cs_num = kwargs['cs_num']
        self.max_cs_len = kwargs['max_cs_len']
        self.max_qa_len  = kwargs['max_qa_len']

        # modules
        self.bert = BertModel(config)
        self.cross_att = AttentionLayer(config.hidden_size, self.cs_num)

        self.cs_merge = AttentionMerge(config.hidden_size, config.hidden_size//2)
        self.qu_merge = AttentionMerge(config.hidden_size, config.hidden_size//2)
        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 3, 1)
        )

        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)    
        return loss, right_num

    def _forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # [B, 5, L] => [B * 5, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.bert(
            input_ids = flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )

        pooler_output = outputs.pooler_output   # outputs[1]  CLS token    [5b, hidden]
        last_hidden_state = outputs.last_hidden_state   # outputs[0]     [5b, seq_len, hidden] 
        # separate query and commonsense encoding        
        # [C] Q [S] QC [S] C [S] cs_1 [S] ←cs_seq_len cs2 ...[S]

        cs_encoding, cs_padding_mask, qa_encoding, qa_padding_mask = self._pad_qacs_to_maxlen(flat_input_ids, last_hidden_state)
        # import pdb; pdb.set_trace()

        # cross-attn
        # [5b, cs_len, query_seq_len, H]
        qc_attn_output, qc_attn_weights = self.cross_att(qa_encoding, cs_encoding, cs_padding_mask)
        # [5b, cs_len, cs_seq_len, H]
        cq_attn_output, cq_attn_weights = self.cross_att(cs_encoding, qa_encoding, qa_padding_mask)

        # [5b, cs_len, cs_seq_len, hidden] => [5b, cs_seq_len, hidden]
        # [5b, cs_seq_len, hidden] => [5b, hidden]
        cs_rep = self.cs_merge(cq_attn_output)
        # cs_rep = torch.mean(cq_attoutput,dim = -3)
        cs_rep = torch.mean(cs_rep, dim = -2)

        # mean pooling query encoding
        qu_rep = self.qu_merge(qc_attn_output)
        # qu_rep = torch.mean(qc_attoutput, dim = -3)
        qu_rep = torch.mean(qu_rep, dim = -2)

        final_rep = torch.cat((pooler_output,cs_rep,qu_rep),dim = -1)
        logits = self.scorer(final_rep).view(-1, 5)

        return logits

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
        sep_ids = input_ids == 102    # sep toekn in bert is 102
        sep_locate = [[] for _ in range(len(sep_ids))]  # [B*5, seq_num]
        for index_1, case in enumerate(sep_ids):
            for index_2, token in enumerate(case):
                if token:
                    sep_locate[index_1].append(index_2)
        # Get CS, QA range
        cs_range_list = [[] for _ in range(len(sep_ids))]   # [B*5, cs_num]
        qa_range_list = []
        for index, case in enumerate(sep_locate):
            # Q [S] QC [S] Choice [S] cs_1[S] cs_2[S]    
            # qa: Q [S] QC [S] Choice [S]; cs: cs_1[S]
            qa_range_list.append(case[2]+1)
            start = case[2]
            for end in case[3:]:
                cs_tuple = (start+1, end+1)
                start = end
                cs_range_list[index].append(cs_tuple)

        # Get CS and stack to tensor
        hidden_size = last_hidden_state.shape[-1]
        cs_batch_list, cs_padding_batch_list = [],[]
        for index, case in enumerate(cs_range_list):
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
        for index, case in enumerate(qa_range_list):
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
        qa_encoding = qa_encoding.unsqueeze(1).expand(-1, self.cs_num, -1, -1)
        qa_padding_mask = torch.stack(qa_padding_batch_list)
        qa_padding_mask = qa_padding_mask.unsqueeze(1).expand(-1, self.cs_num, -1)

        return cs_encoding, cs_padding_mask, qa_encoding, qa_padding_mask


class AttentionMerge(nn.Module):

    def __init__(self, input_size, attention_size, dropout_prob=0.1):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        H (B, L, hidden_size) => h (B, hidden_size)
        (B, L1, L2, hidden_size) => (B, L2, hidden)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.
        # values [batch*5, len, hidden] => keys [B, L, atten_size]
        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)      # variance
        # (b, l, atten_size) @ (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)

        attention_probs = F.softmax(attention_probs * mask, dim=1)  # [batch*5, len, 1]
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)    # [batch*5, hidden]
        return context


class AttentionLayer(nn.Module):
    
    def __init__(self, hidden_size, cs_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.cs_num = cs_num
        self.mult_attn = nn.MultiheadAttention(self.hidden_size, num_heads=1)

    def forward(self, query, keyvalue, attn_mask):
        '''
        input:
        - query: [b, cs_num, Lq, hidden]
        - keyvalue: [b, cs_num, Lkv, hidden]
        
        output:
        - attn_output_weights: [B, cs_num, Lq, Lkv]
        - attn_output: [B, cs_num, Lq, H]
        '''
        q_origin_shape = query.shape
        # [B, cs_num, L, H] -> [B * cs_num, L, H] -> [L, B*cs_num, H]
        query = query.contiguous().view(-1, query.size(-2), query.size(-1))
        query = query.transpose(0, 1)
        keyvalue = keyvalue.contiguous().view(-1, keyvalue.size(-2), keyvalue.size(-1))
        keyvalue = keyvalue.transpose(0, 1)
        
        # [B, cs_num, L] -> [B*cs_num, L]
        attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-1))

        # [Lq, B*cs_num, H], [B*cs_num, Lq, Ls]
        attn_output, attn_output_weights = self.mult_attn(query, keyvalue, keyvalue, key_padding_mask=attn_mask)
        
        # [Lq, B*cs_num, H] -> [B*cs_num, Lq, H] -> [B, cs_num, Lq, H]
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.view(q_origin_shape)

        attn_output_weights = attn_output_weights.view(q_origin_shape[0], self.cs_num, -1)
        return attn_output, attn_output_weights

