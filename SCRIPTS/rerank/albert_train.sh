python CODE/run_csqa_task.py\
    --task_name OMCSrerank_Albert_AttnRanker\
    --mission train\
    --fp16 0\
    --gpu_ids 2\
    --save_mode step\
    --print_step 500\
    --evltest_batch_size 12\
    --eval_after_tacc 0.58\
    \
    --cs_num 2\
    --max_seq_len 78\
    --OMCS_version 1\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 8\
    --learning_rate 2e-5\
    --num_train_epochs 6\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2\

