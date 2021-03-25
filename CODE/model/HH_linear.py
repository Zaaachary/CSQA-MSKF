import pdb
from utils import common


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
from transformers import AlbertPreTrainedModel, AlbertModel, BertPreTrainedModel, BertModel


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
        self.cs_seq_len = kwargs['max_cs_len']
        self.qa_seq_len  = kwargs['max_qa_len']

        # modules
        self.albert = AlbertModel(config)
        # TODO
        self.cross_att = AttentionLayer(config)

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
        import pdb; pdb.set_trace()

        #TODO

        cs_encoding_list = []
        for i in range(self.cs_num):
            start = self.qa_seq_len + i*self.cs_seq_len
            end = start + self.cs_seq_len
            cs_encoding_list.append(last_hidden_state[:,start:end,:])
        cs_encoding_stack = torch.stack(cs_encoding_list, dim=1)

        qa_encoding = last_hidden_state[:,:self.qa_seq_len,:]
        qa_encoding = qa_encoding.unsqueeze(1) # [5B, 1, qa_len, H]
        qa_encoding_expand = qa_encoding.expand(-1, self.cs_num, -1, -1)

        # cross-attn
        # [5b, cs_len, query_seq_len, hidden]
        qc_attoutput, _ = self.cross_att(qa_encoding_expand, cs_encoding_stack)
        # [5b, cs_len, cs_seq_len, hidden]
        cq_attoutput, _ = self.cross_att(cs_encoding_stack, qa_encoding_expand)

        # [5b, cs_len, cs_seq_len, hidden] => [5b, cs_seq_len, hidden]
        # [5b, cs_seq_len, hidden] => [5b, hidden]
        cs_rep = self.cs_merge(cq_attoutput)
        # cs_rep = torch.mean(cq_attoutput,dim = -3)
        cs_rep = torch.mean(cs_rep, dim = -2)

        # mean pooling query encoding
        qu_rep = self.qu_merge(qc_attoutput)
        # qu_rep = torch.mean(qc_attoutput, dim = -3)
        qu_rep = torch.mean(qu_rep, dim = -2)

        final_rep = torch.cat((pooler_output,cs_rep,qu_rep),dim = -1)
        logits = self.scorer(final_rep).view(-1, 5)

        return logits


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
        self.cs_seq_len = kwargs['max_cs_len']
        self.qa_seq_len  = kwargs['max_qa_len']

        # modules
        self.bert = BertModel(config)
        self.cross_att = AttentionLayer(config)
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
        # [C] Q [S] QC [S] C [S] PADDING  ←qa_seq_len  
        # cs_1 PADDING [S] ←cs_seq_len cs2 PADDING ... [S]

        cs_encoding_list = []
        for i in range(self.cs_num):
            start = self.qa_seq_len + i*self.cs_seq_len
            end = start + self.cs_seq_len
            cs_encoding_list.append(last_hidden_state[:,start:end,:])
        cs_encoding_stack = torch.stack(cs_encoding_list, dim=1)

        qa_encoding = last_hidden_state[:,:self.qa_seq_len,:]
        # import pdb; pdb.set_trace()
        qa_encoding = qa_encoding.unsqueeze(1) # [5B, 1, qa_len, H]
        qa_encoding_expand = qa_encoding.expand(-1, self.cs_num, -1, -1)

        # cross-attn
        # [5b, cs_len, query_seq_len, hidden]
        qc_attoutput, _ = self.cross_att(qa_encoding_expand, cs_encoding_stack)
        # [5b, cs_len, cs_seq_len, hidden]
        cq_attoutput, _ = self.cross_att(cs_encoding_stack, qa_encoding_expand)

        # [5b, cs_len, cs_seq_len, hidden] => [5b, cs_seq_len, hidden]
        # [5b, cs_seq_len, hidden] => [5b, hidden]
        cs_rep = self.cs_merge(cq_attoutput)
        # cs_rep = torch.mean(cq_attoutput,dim = -3)
        cs_rep = torch.mean(cs_rep, dim = -2)

        # mean pooling query encoding
        qu_rep = self.qu_merge(qc_attoutput)
        # qu_rep = torch.mean(qc_attoutput, dim = -3)
        qu_rep = torch.mean(qu_rep, dim = -2)

        final_rep = torch.cat((pooler_output,cs_rep,qu_rep),dim = -1)
        logits = self.scorer(final_rep).view(-1, 5)

        return logits


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
    def __init__(self,config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size,config.hidden_size)
        self.value = nn.Linear(config.hidden_size,config.hidden_size)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,query,source):
        '''
        input:
        - query: [b, cs_len, L1, hidden]
        - source: [b, cs_len, L2, hidden]
        
        output:
        - attention_scores: [b, cs_len, L2]     <= sum of attention_prob [b, cs_len, L1, L2]
        - context_layer: [b, cs_len, L1, hidden]
        '''
        # hidden_states should be (batch_size,document_length,hidden_size)
        query_layer = self.query(query)
        key_layer = self.key(source)
        value_layer = self.value(source)
        
        attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Q @ K
        # attention_probs /= query.shape[-1]**(1/2)   # scale
        attention_probs = nn.Softmax(dim=-1)(attention_probs)   # softmax [b, cs_len, L1, L2]
        attention_scores = torch.sum(attention_probs,dim = -2)  # [b, cs_len, L2]
        # attention_scores = torch.sum(attention_probs,dim = -1)  # [b, cs_len, L1]
        context_layer = torch.matmul(attention_probs, value_layer)  # softmax @ V [b, cs_len, L1, hidden]
        return context_layer,attention_scores