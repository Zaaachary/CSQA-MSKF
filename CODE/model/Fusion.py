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

from model.AlbertBurger import AttentionMerge


from utils import common


class MultiSourceFusion(nn.Module):

    def __init__(self, model_list, hidden_size=768):
        super().__init__()

        self.model_num = len(model_list)
        self.hidden_size = hidden_size
        self.activation = nn.ReLU()

        self.fusion = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size*self.model_num, self.hidden_size*self.model_num),
            self.activation,
            
            nn.Linear(self.hidden_size*self.model_num, self.hidden_size),
            self.activation,
        )

        self.scorer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, 1)
        )

        self.apply(self.init_weights)

    def forward(self, *batch):
        labels = batch[-1]
        pooler = batch[:-1]
        logits = self._forward(*pooler)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)    
        return loss, right_num

    def _forward(self, *pooler):
        # pooler [B, 5, H] *n -> [B, 5, nH]
        pooler = torch.cat(pooler, dim=-1)
        
        fuse = self.fusion(pooler)
        fuse = self.activation(fuse)

        logits = self.scorer(fuse)
        logits = logits.squeeze(-1)
        return logits

    def predict(self, *pooler):
        """
        return: [B, 5]
        """
        logits = self._forward(*pooler)
        logits = F.softmax(logits, dim=1)
        return logits


    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

class MultiSourceFusionPlus(nn.Module):

    def __init__(self, model_list, hidden_size=768):
        super().__init__()

        self.model_num = len(model_list)
        self.hidden_size = hidden_size
        self.activation = nn.ReLU()

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.drop_out = nn.Dropout(0.1)

        self.ffn = nn.Linear(self.hidden_size*self.model_num, self.hidden_size*self.model_num)
        self.ffn_output = nn.Linear(self.hidden_size*self.model_num, self.hidden_size)
        self.scorer = nn.Linear(self.hidden_size, 1)

        self.apply(self.init_weights)

    def forward(self, *batch):
        labels = batch[-1]
        pooler = batch[:-1]
        logits = self._forward(*pooler)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)    
        return loss, right_num

    def _forward(self, *pooler):
        # pooler [B, 5, H] *n -> [B, 5, nH]
        pooler = torch.cat(pooler, dim=-1)
        pooler = self.layer_norm(pooler)
        pooler = self.drop_out(pooler)
        
        interffn = self.activation(self.ffn(pooler))
        ffn_output = self.ffn_output(pooler + interffn) # 残差
        ffn_output = self.activation(ffn_output)
        ffn_output = self.layer_norm(ffn_output)

        logits = self.scorer(self.drop_out(ffn_output))
        logits = logits.squeeze(-1)
        return logits

    def predict(self, *pooler):
        """
        return: [B, 5]
        """
        logits = self._forward(*pooler)
        logits = F.softmax(logits, dim=1)
        return logits

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

class MultiSourceAttnMerge(nn.Module):

    def __init__(self, model_list, hidden_size=768):
        super().__init__()

        self.model_num = len(model_list)
        self.hidden_size = hidden_size
        self.activation = nn.ReLU()

        self.fusion = AttentionMerge(self.hidden_size, self.hidden_size//2, 0.1)

        self.scorer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, 1)
        )

        self.apply(self.init_weights)

    def forward(self, *batch):
        labels = batch[-1]
        pooler = batch[:-1]
        logits = self._forward(*pooler)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)    
        return loss, right_num

    def _forward(self, *pooler):
        # pooler [B, 5, H] *n -> [B, 5, nH]
        # import pdb; pdb.set_trace()
        pooler = torch.stack(pooler, dim=2)
        fuse = self.fusion(pooler)
        fuse = self.activation(fuse)

        logits = self.scorer(fuse)
        logits = logits.squeeze(-1)
        return logits

    def predict(self, *pooler):
        """
        return: [B, 5]
        """
        logits = self._forward(*pooler)
        logits = F.softmax(logits, dim=1)
        return logits

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


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
        # import pdb; pdb.set_trace()
        attention_probs = F.softmax(attention_probs * mask, dim=-2)  # [batch*5, len, 1]
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=-2)    # [batch*5, hidden]
        return context
