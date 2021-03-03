#! -*- encoding:utf-8 -*-
"""
@File    :   Headhunter.py
@Author  :   
@Contact :   
@Dscpt   :   
"""

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AlbertModel, AlbertPreTrainedModel, BertModel,
                          BertPreTrainedModel)
from transformers.modeling_utils import SequenceSummary


class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size,config.hidden_size)
        self.value = nn.Linear(config.hidden_size,config.hidden_size)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,hidden_states):
        # hidden_states should be (batch_size,document_length,hidden_size)
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_probs)
        attention_scores = torch.sum(attention_probs,dim = -2)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer,attention_scores


class BertAttRanker(BertPreTrainedModel):
    def __init__(self, config, cs_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.bert = BertModel(config)
        self.self_att = SelfAttention(config)
        self.classifier = nn.Linear(config.hidden_size,1)
        # self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
    ):
        batch_size,input_size = input_ids.shape[:2]
        # pdb.set_trace()
        num_choices = int(input_size/self.cs_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = bert_outputs[1]
        reshaped_output = pooled_output.view(int(batch_size*num_choices),self.cs_len,pooled_output.size(-1))

        atten_output,attention_scores = self.self_att(reshaped_output)
        attention_scores = attention_scores.view(batch_size,num_choices,-1)
        # attention summary 
        atten_output = atten_output.view(batch_size,num_choices,self.cs_len,-1)
        attention_scores = F.softmax(attention_scores,dim = -1).unsqueeze(2)
        atten_output = torch.tanh(torch.matmul(attention_scores,atten_output)).squeeze(2)

        # reshaped_output = atten_output.view(int(batch_size*num_choices),self.cs_len*atten_output.size(-1))

        logits = self.classifier(atten_output)
        reshaped_logits = logits.view(-1, num_choices)
        
        outputs = (reshaped_logits,attention_scores)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs

