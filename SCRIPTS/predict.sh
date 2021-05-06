python CODE/run_csqa_task.py\
    --task_name WKDT_Albert_Baseline\
    --mission predict\
    --fp16 0\
    --gpu_ids 3\
    --save_mode step\
    --print_step 100\
    --eval_after_tacc 0\
    --evltest_batch_size 12\
    \
    --max_seq_len 130\
    --max_desc_len 40\
    --max_qa_len 54\
    --WKDT_version 4.0\
    \
    --train_batch_size 0\
    --gradient_accumulation_steps 0\
    --learning_rate 0\
    --num_train_epochs 0\
    --warmup_proportion 0\
    --weight_decay 0\
    \
    --dataset_dir DATA\
    --saved_model_dir /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1829-May04_seed5004_wkdtv4.0/\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2/
