python CODE/run_csrk_task.py\
    --task_name RankOMCS_Albert\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 0\
    --save_mode step\
    --print_step 500\
    --evltest_batch_size 64\
    --eval_after_tacc 0.70\
    \
    --max_seq_len 80\
    --CSRK_version 0.3\
    --split_method topbotton2\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 16\
    --learning_rate 2e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/albert-xxlarge-v2
 
