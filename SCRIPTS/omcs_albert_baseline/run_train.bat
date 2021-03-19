python CODE\\run_task.py^
    --task_name OMCS_Albert_Baseline^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 100^
    ^
    --cs_num 4^
    --max_seq_len 128^
    --train_batch_size 2^
    --evltest_batch_size 12^
    --gradient_accumulation_steps 16^
    --lr 2e-5^
    --num_train_epochs 2^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^