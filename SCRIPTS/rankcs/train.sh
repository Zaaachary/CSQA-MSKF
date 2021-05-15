python CODE/run_csrk_task.py\
    --task_name RankOMCS_Bert\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 6\
    --save_mode step\
    --print_step 25\
    --evltest_batch_size 12\
    --eval_after_tacc 0.60\
    \
    --max_seq_len 80\
    --CSRK_version 0.1\
    --split_method half\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5\
    --num_train_epochs 10\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir bert-base-uncased
 
