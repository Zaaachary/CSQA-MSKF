python CODE/run_csqa_task.py\
    --task_name Origin_Bert_Baseline\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 0\
    --save_mode step\
    --print_step 100\
    --evltest_batch_size 12\
    --eval_after_tacc 0.60\
    \
    --max_seq_len 80\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5\
    --num_train_epochs 4\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/bert-base-cased
 
