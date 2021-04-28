python CODE/run_dapt_task.py\
    --task_name OMCS_BertMLM\
    --mission train\
    --fp16 0\
    --gpu_ids 1\
    --seed 42\
    --save_mode epoch\
    --print_step 200\
    --evltest_batch_size 12\
    \
    --mask_pct 0.20\
    --max_seq_len 35\
    --mask_method random\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --num_train_epochs 5\
    --learning_rate 3e-5\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/bert-base-cased\
    --saved_model_dir none
