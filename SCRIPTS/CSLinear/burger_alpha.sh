python CODE/run_task.py\
    --task_name CSLinear_Albert_BurgerAlpha4\
    --mission train\
    --fp16 0\
    --gpu_ids 5\
    --save_mode step\
    --print_step 100\
    --eval_after_tacc 0.59\
    --evltest_batch_size 12\
    \
    --cs_num 3\
    --max_qa_len 54\
    --max_cs_len 20\
    --max_seq_len 140\
    --OMCS_version 1\
    --albert1_layers 10\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2\
