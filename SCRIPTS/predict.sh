python CODE/run_csqa_task.py\
    --task_name OMCS_Albert_Baseline\
    --mission predict\
    --fp16 0\
    --gpu_ids 0\
    --save_mode step\
    --print_step 100\
    --eval_after_tacc 0\
    --evltest_batch_size 16\
    \
    --max_seq_len 100\
    --OMCS_version 3.0\
    \
    --train_batch_size 2\
    --gradient_accumulation_steps 4\
    --learning_rate 2e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  DATA/result/\
    --saved_model_dir /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1312-Apr26_seed42_cs2_omcsv3.0/\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2/
