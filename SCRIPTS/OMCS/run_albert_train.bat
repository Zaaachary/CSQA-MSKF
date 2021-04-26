python CODE\\run_task.py^
    --task_name Origin_Albert_Baseline^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 50^
    --eval_after_tacc 0^
    --evltest_batch_size 12^
    ^
    --max_seq_len 140^
    --OMCS_version 3.0^
    --cs_num 4^
    ^
    --train_batch_size 4^
    --gradient_accumulation_steps 4^
    --learning_rate 2e-5^
    --num_train_epochs 10^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^