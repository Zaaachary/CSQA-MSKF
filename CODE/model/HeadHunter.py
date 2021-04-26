#! -*- encoding:utf-8 -*-
"""
@File    :   Headhunter.py
@Author  :   
@Contact :   
@Dscpt   :   
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertPreTrainedModel


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


class SequenceSummaryLayer(nn.Module):
    def __init__(self,hidden_size,summary_layers):
        super().__init__()
        self.summary_layers = summary_layers
        self.linear = nn.Linear(hidden_size * summary_layers, hidden_size)
        # do pooler just as transformers did
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, x):
        stacked_hidden_states = torch.stack(list(x[-self.summary_layers:]),dim = -2)
        # print(stacked_hidden_states.shape)
        stacked_hidden_states = stacked_hidden_states[:,0]
        # pdb.set_trace()
        concat_hidden_states = stacked_hidden_states.view(stacked_hidden_states.shape[0],stacked_hidden_states.shape[-2]*stacked_hidden_states.shape[-1])
        resized_hidden_states = self.linear(concat_hidden_states)
        pooled_hidden_states = self.pooler_activation(self.pooler(resized_hidden_states))
        return pooled_hidden_states


class AlbertAttRanker(AlbertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.cs_len = kwargs['cs_num']
        self.albert = AlbertModel(config)
        self.sequence_summary = SequenceSummaryLayer(config.hidden_size,4)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.self_att = SelfAttention(config)
        self.classifier = nn.Linear(config.hidden_size,1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        batch_size, input_size = input_ids.shape[:2]
        num_choices = int(input_size/self.cs_len)

        # [B, 5*cs_num, seq_len] -> [B5cs_num, seq_len]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        # import pdb; pdb.set_trace()
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states = True
        )
        hidden_output = outputs[2]
        pooled_output = self.sequence_summary(outputs[2])
        pooled_output = self.dropout(pooled_output)
        # pooled_output [5B, cs_len, Hidden]
        reshaped_output = pooled_output.view(int(batch_size*num_choices),self.cs_len,pooled_output.size(-1))

        # [5B, cs_len, Hidden], [5B, cs_len]
        atten_output,attention_scores = self.self_att(reshaped_output)
        # [B, 5, cs_len], [B, 5, cs_len, Hidden]
        attention_scores = attention_scores.view(batch_size,num_choices,-1)
        atten_output = atten_output.view(batch_size,num_choices,self.cs_len,-1)
        # attention summary 
        # [B, 5, 1, cs_len]
        attention_scores = F.softmax(attention_scores,dim = -1).unsqueeze(2)
        # [B, Hidden]
        atten_output = torch.tanh(torch.matmul(attention_scores,atten_output)).squeeze(2)
        
        logits = self.classifier(atten_output)
        reshaped_logits = logits.view(-1, num_choices)
        
        outputs = (reshaped_logits, attention_scores)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

            with torch.no_grad():
                logits = F.softmax(logits, dim=1)       # get the score
                predicts = torch.argmax(logits, dim=1)  # find the result
                right_num = torch.sum(predicts == labels)

        return loss, right_num
        # return outputs

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 5]
        """
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        logits = F.softmax(logits, dim=1)
        return logits