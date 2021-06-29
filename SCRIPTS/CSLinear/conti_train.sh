python CODE/run_task.py\
    --task_name CSLinear_Albert_BurgerAlpha2\
    --mission conti-train\
    --fp16 0\
    --gpu_ids 5\
    --save_mode step\
    --print_step 100\
    --eval_after_tacc 0\
    --evltest_batch_size 12\
    --clip_batch_off\
    \
    --cs_num 4\
    --max_qa_len 54\
    --max_cs_len 20\
    --max_seq_len 140\
    \
    --train_batch_size 2\
    --albert1_layers 10\
    --gradient_accumulation_steps 4\
    --learning_rate 2e-5\
    --num_train_epochs 2\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir DATA\
    --result_dir /data/zhifli/model_save \
    --saved_model_dir /data/zhifli/model_save/albert-base-v2/CSLinear_Albert_BurgerAlpha2/0952-Apr07_seed42_cs4_layer10 \
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2\

