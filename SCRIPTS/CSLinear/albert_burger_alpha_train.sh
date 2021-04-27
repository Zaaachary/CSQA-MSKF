python CODE/run_csqa_task.py\
    --task_name CSLinear_Albert_BurgerAlpha2\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 2\
    --save_mode step\
    --print_step 500\
    --eval_after_tacc 0.80\
    --evltest_batch_size 12\
    \
    --cs_num 1\
    --max_qa_len 54\
    --max_cs_len 20\
    --max_seq_len 78\
    --OMCS_version 3.0\
    --albert1_layers 8\
    \
    --train_batch_size 1\
    --gradient_accumulation_steps 8\
    --learning_rate 2e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2\

