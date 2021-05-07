python CODE/run_csqa_task.py\
    --task_name MSKE_Albert_Baseline\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 7\
    --save_mode step\
    --print_step 100\
    --evltest_batch_size 12\
    --eval_after_tacc 0.60\
    --dev_method train_01_equal\
    \
    --OMCS_version 3.0\
    --WKDT_version 4.0\
    --max_seq_len 130\
    --cs_num 4\
    --train_method train_01_equal\
    \
    --train_batch_size 5\
    --gradient_accumulation_steps 1\
    --learning_rate 5e-6\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-base-v2