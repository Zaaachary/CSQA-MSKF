python CODE/run_csqa_task.py\ 
    --task_name OMCS_Albert_Baseline\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 0\
    --save_mode step\
    --print_step 100\
    --evltest_batch_size 16\
    --eval_after_tacc 0.59\
    \
    --max_seq_len 140\
    --OMCS_version 3.0_rank\
    --cs_num 5\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 1e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2
 
