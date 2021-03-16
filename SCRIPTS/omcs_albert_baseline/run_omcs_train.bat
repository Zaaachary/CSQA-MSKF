python CODE\\run_task.py^
    --task_name OMCS_Albert_Baseline^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --print_step 100^
    ^
    --cs_num 0^
    --train_batch_size 4^
    --evltest_batch_size 12^
    --gradient_accumulation_steps 8^
    --lr 2e-6^
    --num_train_epochs 10^
    --warmup_proportion 0.2^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --pred_file_dir  DATA/result/task_result.json^
    --model_save_dir DATA/result/albert-base-v2/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^