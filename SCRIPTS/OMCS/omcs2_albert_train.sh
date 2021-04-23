python CODE/run_task.py\
    --task_name OMCS_Albert_Baseline\
    --mission train\
    --fp16 0\
    --gpu_ids 5\
    --save_mode step\
    --seed 42\
    --print_step 200\
    --evltest_batch_size 12\
    --eval_after_tacc 0.58\
    \
    --cs_num 4\
    --max_seq_len 140\
    --OMCS_version 3.0\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5\
    --num_train_epochs 6\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2\
