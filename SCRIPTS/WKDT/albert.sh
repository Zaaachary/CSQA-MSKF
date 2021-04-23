python CODE/run_task.py\
    --task_name WKDT_Albert_Baseline\
    --mission train\
    --fp16 0\
    --gpu_ids 1\
    --save_mode step\
    --print_step 400\
    --eval_after_tacc 0.79\
    --evltest_batch_size 12\
    \
    --max_seq_len 130\
    --max_qa_len 58\
    --max_desc_len 40\
    --WKDT_version 3.0\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 8\
    --learning_rate 2e-5\
    --num_train_epochs 10\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save/\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2
