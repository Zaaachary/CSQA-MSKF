python CODE/run_csqa_task.py\
    --task_name CSLE_Albert_BurgerAlpha2\
    --mission train\
    --fp16 0\
    --gpu_ids 6\
    --save_mode step\
    --print_step 100\
    --eval_after_tacc 0.59\
    --evltest_batch_size 12\
    --dev_method 0123\
    \
    --train_method train_01\
    --cs_num 8\
    --model_cs_num 4\
    --max_qa_len 54\
    --max_cs_len 20\
    --max_seq_len 140\
    --OMCS_version 3.0\
    --albert1_layers 10\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5\
    --num_train_epochs 6\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2\

