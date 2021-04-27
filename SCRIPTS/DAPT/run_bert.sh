python CODE/run_dapt_task.py\
    --task_name Webster_Bert\
    --mission train\
    --fp16 0\
    --gpu_ids 4\
    --seed 42\
    --save_mode step\
    --print_step 200\
    --evltest_batch_size 12\
    --eval_after_tacc 0.85\
    \
    --DAPT_version 1.0\
    --mask_pct 0.20\
    --max_seq_len 45\
    --mask_method random\
    \
    --train_batch_size 16\
    --gradient_accumulation_steps 1\
    --num_train_epochs 1\
    --learning_rate 2e-5\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/bert-base-cased\
    --saved_model_dir none
