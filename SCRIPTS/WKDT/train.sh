python CODE/run_csqa_task.py\
    --task_name WKDT_Albert_Baseline\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 5\
    --save_mode step\
    --print_step 500\
    --evltest_batch_size 12\
    --eval_after_tacc 0.79\
    \
    --max_seq_len 130\
    --max_qa_len 54\
    --max_desc_len 40\
    --WKDT_version 4.0\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 16\
    --learning_rate 1e-5\
    --num_train_epochs 10\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-xxlarge-v2
