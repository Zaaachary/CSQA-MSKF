#! -*- encoding:utf-8 -*-
"""
@File    :   models.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertPreTrainedModel


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

        attention_probs = F.softmax(attention_probs * mask, dim=1)  # [batch*5, len, 1]
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)    # [batch*5, hidden]
        return context


class AlbertAttnMerge(AlbertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(AlbertAttnMerge, self).__init__(config)

        self.albert = AlbertModel(config)

        self.att_merge = AttentionMerge(
            config.hidden_size, 1024, 0.1)

        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        """
        input_ids: [B, 5, L]
        labels: [B, ]
        """
        # logits: [B, 2]
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)      # get the CELoss

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)       # get the score
            predicts = torch.argmax(logits, dim=1)  # find the result
            right_num = torch.sum(predicts == labels)

        return loss, right_num

    def _forward(self, input_ids, attention_mask, token_type_ids):
        # [B, 5, L] => [B * 5, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        
        # outputs[0]: [B*5, L, H] => [B*5, H]
        # flat_attention_mask [B*5, L]
        h12 = self.att_merge(outputs[0], flat_attention_mask)   

        # [B*5, H] => [B*5, 1] => [B, 5]
        logits = self.scorer(h12).view(-1, 5)

        return logits

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 5]
        """
        return self._forward(input_ids, attention_mask, token_type_ids)


class AlbertAddTFM(AlbertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(AlbertAddTFM, self).__init__(config)

        self.albert = AlbertModel(config)

        self.tfm = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=16,
            dim_feedforward=512, dropout=0.1
        )

        self.att_merge = AttentionMerge(
            config.hidden_size, 1024, 0.1)

        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        """
        input_ids: [B, 5, L]
        labels: [B, ]
        """
        # logits: [B, 2]
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)      # get the CELoss

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)       # get the score
            predicts = torch.argmax(logits, dim=1)  # find the result
            right_num = torch.sum(predicts == labels)

        return loss, right_num

    def _forward(self, input_ids, attention_mask, token_type_ids):
        # [B, 5, L] => [B * 5, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )

        # outputs[0]: [B*5, L, H] => output [L, B*5, H] => output [B*5, L, H]
        output = outputs[0].transpose(0, 1)
        mask = flat_attention_mask + 1
        mask = mask.masked_fill(mask==2, 0).bool()
        output = self.tfm(src=output, src_key_padding_mask=mask)
        # output = self.tfm(src=output, src_key_padding_mask=mask)
        output = output.transpose(0, 1)

        # output [B*5, L, H] => [B*5, H]
        h12 = self.att_merge(output, flat_attention_mask)   

        # [B*5, H] => [B*5, 1] => [B, 5]
        logits = self.scorer(h12).view(-1, 5)

        return logits

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 5]
        """
        return self._forward(input_ids, attention_mask, token_type_ids)

