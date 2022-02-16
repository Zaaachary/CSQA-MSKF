# Model for Commonsense QA

## Albert Baseline
description https://github.com/Zaaachary/CSQA-Description/blob/main/Albert-model-desc.md

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
