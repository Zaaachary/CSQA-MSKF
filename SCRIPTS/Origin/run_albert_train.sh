python CODE/run_task.py\
    --task_name Origin_Albert_Baseline\
    --mission train\
    --fp16 0\
    --gpu_ids 2\
    --save_mode step\
    --print_step 100\
    --evltest_batch_size 12\
    --eval_after_tacc 0.8\
    \
    --max_seq_len 140\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 16\
    --learning_rate 2e-5\
    --num_train_epochs 10\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-xxlarge-v2\
 
