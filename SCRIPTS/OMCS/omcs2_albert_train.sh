python CODE/run_csqa_task.py\
    --task_name OMCS_Albert_Baseline\
    --mission train\
    --fp16 0\
    --gpu_ids 0\
    --save_mode step\
    --seed 42\
    --print_step 500\
    --evltest_batch_size 8\
    --eval_after_tacc 0.80\
    \
    --cs_num 2\
    --max_seq_len 100\
    --OMCS_version 3.0\
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
