python CODE/run_dapt_task.py\
    --task_name Webster_BertPT\
    --mission train\
    --fp16 0\
    --gpu_ids 4\
    --seed 42\
    --save_mode epoch\
    --print_step 50\
    --evltest_batch_size 12\
    \
    --DAPT_version 1.0\
    --mask_pct 0.20\
    --max_seq_len 40\
    --mask_method random\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 2\
    --num_train_epochs 5\
    --learning_rate 2e-5\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/bert-base-cased\
    --saved_model_dir none
