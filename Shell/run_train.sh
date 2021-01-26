python3 run_task.py\
    --batch_size 2\
    --gpu_ids -1\
    --lr 1e-5\
    --num_train_epochs 1\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    --fp16 0\
    --print_step 100\
    --mission train\
    --train_file_name DATA/csqa/train_data.json\
    --dev_file_name DATA/csqa/dev_data.json\
    --test_file_name DATA/csqa/trial_data.json\
    --pred_file_name  DATA/result/task_result.json\
    --output_model_dir DATA/result/model/\
    --pretrained_model_dir albert-large-v2\
    --pretrained_vocab_dir albert-large-v2