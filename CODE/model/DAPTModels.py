#! -*- encoding:utf-8 -*-
"""
@File    :   DAPTModels.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel

from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertOnlyMLMHead


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sequence_labels=None,
        desc_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if sequence_labels is not None and desc_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), sequence_labels.view(-1))
            right_desc_loss = loss_fct(seq_relationship_score.view(-1, 2), desc_labels.view(-1))
            total_loss = masked_lm_loss + right_desc_loss
            
            with torch.no_grad():
                logits = F.softmax(seq_relationship_score, dim =1)
                predicts = torch.argmax(logits, dim=1)
                right_num = torch.sum(predicts == desc_labels)

        return (total_loss, masked_lm_loss, right_desc_loss, right_num) if total_loss is not None else outputs


class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sequence_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if sequence_labels is not None:
            loss_fct = CrossEntropyLoss()    # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), sequence_labels.view(-1))

        return (masked_lm_loss, masked_lm_loss) if masked_lm_loss is not None else outputs
