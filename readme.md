```shell
python task.py --batch_size 4 --lr 1e-5 --num_train_epochs 1 --warmup_proportion 0.1 --weight_decay 0.1 --fp16 0 --train_file_name csqa_data\conceptnet\weight_rel\train_data.json --devlp_file_name csqa_data\conceptnet\weight_rel\dev_data.json --trial_file_name csqa_data\conceptnet\weight_rel\trial_data.json --pred_file_name  csqa_data\task_result.json --output_model_dir csqa_data\conceptnet\weight_rel --bert_model_dir albert-base-v2 --bert_vocab_dir albert-base-v2 --print_step 100 --mission train
```

```  
(albert): AlbertModel(
    (embeddings): AlbertEmbeddings(
      (word_embeddings): Embedding(30000, 128, padding_idx=0)
      (position_embeddings): Embedding(512, 128)
      (token_type_embeddings): Embedding(2, 128)
      (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0, inplace=False)
    )
    (encoder): AlbertTransformer(
      (embedding_hidden_mapping_in): Linear(in_features=128, out_features=1024, bias=True)
      (albert_layer_groups): ModuleList(
        (0): AlbertLayerGroup(
          (albert_layers): ModuleList(
            (0): AlbertLayer(
              (full_layer_layer_norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
              (attention): AlbertAttention(
                (query): Linear(in_features=1024, out_features=1024, bias=True)
                (key): Linear(in_features=1024, out_features=1024, bias=True)
                (value): Linear(in_features=1024, out_features=1024, bias=True)
                (dropout): Dropout(p=0, inplace=False)
                (dense): Linear(in_features=1024, out_features=1024, bias=True)
                (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
              )
              (ffn): Linear(in_features=1024, out_features=4096, bias=True)
              (ffn_output): Linear(in_features=4096, out_features=1024, bias=True)
            )
          )
        )
      )
    )
    (pooler): Linear(in_features=1024, out_features=1024, bias=True)
    (pooler_activation): Tanh()
  )
  (att_merge): AttentionMerge(
    (hidden_layer): Linear(in_features=1024, out_features=1024, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (scorer): Sequential(
    (0): Dropout(p=0.1, inplace=False)
    (1): Linear(in_features=1024, out_features=1, bias=True)
  )
)
```



ALBERT -> Attention Merge -> Scorer

[batch_size, 2, seq_len] --ALBERT-->  [batch_size, ]

