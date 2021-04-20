python CODE\\run_task.py^
    --task_name Origin_Albert_Baseline^
    --mission conti-train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 100^
    --eval_after_tacc 0.58^
    --evltest_batch_size 12^
    ^
    --max_seq_len 128^
    ^
    --train_batch_size 2^
    --gradient_accumulation_steps 4^
    --learning_rate 2e-5^
    --num_train_epochs 8^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --saved_model_dir D:\CODE\Commonsense\CSQA_dev\DATA\result\albert-base-v2\Origin_Albert_Baseline\1253-Apr20_seed42^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^