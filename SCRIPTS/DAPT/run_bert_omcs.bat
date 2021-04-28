python CODE\run_dapt_task.py^
    --task_name OMCS_BertMLM^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --seed 42^
    --save_mode epoch^
    --print_step 10^
    --evltest_batch_size 12^
    ^
    --mask_pct 0.20^
    --max_seq_len 40^
    --mask_method random^
    ^
    --train_batch_size 2^
    --gradient_accumulation_steps 2^
    --num_train_epochs 5^
    --learning_rate 2e-5^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA\result^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\bert-base-cased^
    --saved_model_dir none
