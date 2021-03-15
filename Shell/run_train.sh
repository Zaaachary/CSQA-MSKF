python CODE/run_task.py\
    --task_name AlbertBaseine\
    --mission train\
    --fp16 0\
    --gpu_ids 0\
    --print_step 100\
    \
    --batch_size 4\
    --lr 1e-5\
    --num_train_epochs 4\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --pred_file_dir  ../DATA/result/task_result.json\
    --model_save_dir ../DATA/result/TCmodel/\
    --PTM_model_vocab_dir albert-base-v2\