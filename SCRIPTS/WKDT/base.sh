python CODE/run_csqa_task.py\
    --task_name WKDT_Albert_Baseline\
    --mission train\
    --fp16 0\
    --seed 5017\
    --gpu_ids 6\
    --save_mode step\
    --print_step 100\
    --evltest_batch_size 12\
    --eval_after_tacc 0.56\
    \
    --max_seq_len 140\
    --max_qa_len 58\
    --max_desc_len 40\
    --WKDT_version 5.0_rank\
    \
    --train_batch_size 16\
    --gradient_accumulation_steps 1\
    --learning_rate 1e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2
