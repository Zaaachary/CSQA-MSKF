python CODE/run_csqa_task.py^
    --task_name OMCSrerank_Albert_AttnRanker^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 50^
    --evltest_batch_size 12^
    --eval_after_tacc 0^
    ^
    --cs_num 2^
    --max_seq_len 120^
    --OMCS_version 1^
    ^
    --train_batch_size 2^
    --gradient_accumulation_steps 8^
    --learning_rate 2e-5^
    --num_train_epochs 6^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^