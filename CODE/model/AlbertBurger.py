#! -*- encoding:utf-8 -*-
"""
@File    :   AlbertBurger.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import math
import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AlbertModel as tfms_AlbertModel
from transformers import AlbertPreTrainedModel, AlbertConfig
from .AlbertModel import AlbertModel


from utils import common


class AlbertBurger():

    def __init__(self, config, **kwargs):
        self.cs_num = None
        self.max_cs_len = None
        self.max_qa_len  = None

        config1 = copy.deepcopy(config)
        config1.num_hidden_layers = 6
        config1.without_embedding = False
        
        config2 = copy.deepcopy(config)
        config2.num_hidden_layers = 6
        config2.without_embedding = True


        # modules
        self.albert = AlbertModel(config1)
        self.cs_attention_scorer = AttentionLayer(config, self.cs_num)
        self.albert2 = AlbertModel(config2)
        self.attention_merge = AttentionMerge(config.hidden_size, config.hidden_size//4, 0.1)

        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cls()

    def forward(self):
        pass




class AlbertBurger_alpha(AlbertPreTrainedModel):
    '''
    input_ids [b, 5, seq_len] => [5b, seq_len]
    => PTM
    cs_encoding [5b, cs_num, cs_seq_len, hidden]
    '''
    def __init__(self, config, **kwargs):
        super(AlbertBurger, self).__init__(config)
        # length config
        self.cs_num = kwargs['cs_num']
        self.max_cs_len = kwargs['max_cs_len']
        self.max_qa_len  = kwargs['max_qa_len']

        # modules
        self.albert = AlbertModel(config)
        self.cs_attention_scorer = AttentionLayer(config, self.cs_num)
        # self.albert2 = AlbertModel(config)
        self.attention_merge = AttentionMerge(config.hidden_size, config.hidden_size//4, 0.1)

        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
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
        # pooler_output = outputs.pooler_output   # outputs[1]  [5B, H]
        last_hidden_state = outputs.last_hidden_state   # outputs[0]  [5B, L, H] 

        # separate query and commonsense encoding
        # encoding:[5B, cs_num, L, H]  mask:[5B, cs_num, L]
        cs_encoding, _, qa_encoding, qa_padding_mask = self._pad_qacs_to_maxlen(flat_input_ids, last_hidden_state)
        qa_encoding_expand = qa_encoding.unsqueeze(1).expand(-1, self.cs_num, -1, -1)
        qa_padding_mask_expand = qa_padding_mask.unsqueeze(1).expand(-1, self.cs_num, -1)

        # import pdb; pdb.set_trace()
        # attn_output:[5B, cs_num, L, H] attn_weights:[5B, cs_num, Lq, Lc]
        attn_output, attn_weights = self.cs_attention_scorer(cs_encoding, qa_encoding_expand, qa_padding_mask_expand)
        new_hidden_state = self._remvoe_cs_pad_add_to_last_hidden_state(attn_output, last_hidden_state)

        merge = self.attention_merge(new_hidden_state, flat_attention_mask)

        logits = self.scorer(merge).view(-1,5)

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
            
        
class AttentionLayer(nn.Module):
    
    def __init__(self, config, cs_num):
        super().__init__()
        self.hidden_size = config.hidden_size
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
        Batch_size, cs_num, Lq, hidden_size = query.shape
        Lkv = keyvalue.shape[-2]
        
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
        attn_output = attn_output.view(Batch_size, cs_num, Lq, hidden_size)

        # [B*cs_num, Lq, Lkv] -> [B, cs_num, Lq, Lkv]
        attn_output_weights = attn_output_weights.view(Batch_size, self.cs_num, Lq, Lkv)
        return attn_output, attn_output_weights


class AttentionMerge(nn.Module):

    def __init__(self, input_size, attention_size, dropout_prob):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        H (B, L, hidden_size) => h (B, hidden_size)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.
        # values [batch*5, len, hidden]
        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        # (b, l, h) + (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)
        import pdb; pdb.set_trace()
        attention_probs = F.softmax(attention_probs * mask, dim=1)  # [batch*5, len, 1]
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)    # [batch*5, hidden]
        return context