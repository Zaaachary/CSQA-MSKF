# Model for Commonsense QA

## Requirement
```
pip install pytorch
pip install transformers
pip install SentencePiece
```

## Albert Baseline
[model description](https://github.com/Zaaachary/CSQA-Description/blob/main/Albert-model-desc.md)

```
!python CODE/run_csqa_task.py\
    --task_name Origin_Albert_Baseline\
    --mission train\
    --seed 42\
    --fp16 0\
    --gpu_ids 0\
    --save_mode step\
    --print_step 500\
    --evltest_batch_size 12\
    --eval_after_tacc 0.78\
    --max_seq_len 128\
    --train_batch_size 2\
    --gradient_accumulation_steps 8\
    --learning_rate 2e-5\
    --num_train_epochs 10\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    --dataset_dir DATA\
    --result_dir  [YOUR OUTPUT]\
    --PTM_model_vocab_dir albert-xxlarge-v2\
```

## MSKF Model

Paper: [基于多知识源融合的级联式常识问答方法](https://kns.cnki.net/kcms2/article/abstract?v=3uoqIhG8C44YLTlOAiTRKibYlV5Vjs7iJTKGjg9uTdeTsOI_ra5_XRHMpetUSJMYHytyW09ZL_jzsDV8Na6-sVouHZ1vMQW2&uniplatform=NZKPT)
