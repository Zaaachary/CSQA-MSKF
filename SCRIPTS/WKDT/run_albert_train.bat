python CODE\\run_task.py^
    --task_name WKDT_Albert_Baseline^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 100^
    --eval_after_tacc 0.58^
    --evltest_batch_size 12^
    ^
    --max_seq_len 128^
    --max_qa_len 54^
    --max_desc_len 35^
    --WKDT_version 3.0^
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
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^