python CODE/run_csqa_task.py\
    --task_name MSKE_Albert_Baseline\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 5\
    --save_mode step\
    --print_step 1000\
    --evltest_batch_size 12\
    --eval_after_tacc 0.80\
    --dev_method top2\
    \
    --OMCS_version 3.0\
    --WKDT_version 4.0\
    --max_seq_len 130\
    --cs_num 8\
    --train_method train_01\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 16\
    --learning_rate 5e-6\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2
