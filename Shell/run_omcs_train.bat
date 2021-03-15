python CODE\\run_task.py^
    --task_name Albert_OMCS_Baseline^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --print_step 100^
    ^
    --batch_size 2^
    --gradient_accumulation_steps 8^
    --lr 2e-5^
    --num_train_epochs 4^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    --cs_num 0^
    ^
    --dataset_dir DATA^
    --pred_file_dir  DATA/result/task_result.json^
    --model_save_dir DATA/result/albert-base-v2/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^