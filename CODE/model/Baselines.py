#! -*- encoding:utf-8 -*-
"""
@File    :   Baselines.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertPreTrainedModel, BertPreTrainedModel, BertModel

class AlbertBaseline(AlbertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super(AlbertBaseline, self).__init__(config)
        self.albert = AlbertModel(config)

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
        
        pooler_output = outputs.pooler_output  # [CLS]

        # [B*5, H] => [B*5, 1] => [B, 5]
        logits = self.scorer(pooler_output).view(-1, 5)

        return logits

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 5]
        """
        return self._forward(input_ids, attention_mask, token_type_ids)


class BertBaseline(BertPreTrainedModel):

    def __init__(self, config, *args, **kwargs):
        super(BertBaseline, self).__init__(config)

        self.bert = BertModel(config)

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

        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        
        pooler_output = outputs.pooler_output  # [CLS]

        # [B*5, H] => [B*5, 1] => [B, 5]
        logits = self.scorer(pooler_output).view(-1, 5)

        return logits

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 5]
        """
        return self._forward(input_ids, attention_mask, token_type_ids)