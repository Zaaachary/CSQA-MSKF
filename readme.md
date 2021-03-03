# CSQA


## Model Architecture

### MS-Model

**ALBERT -> Attention Merge -> Scorer**

example: `<idx, input_ids, attention_mask, token_type_ids, labels>`

batch: `[batch_size, example_size]`

#### Global

input_ids `[B, 5, L]` -> flat_input_ids `[B*5, L]` -ALBERT-> ouput[0] `[B*5, L, Hidden_size]` -Attention Merge-> hidden_merged `[B*5, H]` -Scorer-> logits `[B, 5]`

#### Scorer

dropout + linear `[B*5, H]` -linear-> `[B*5, 1]` -> [B, 5]`

## 