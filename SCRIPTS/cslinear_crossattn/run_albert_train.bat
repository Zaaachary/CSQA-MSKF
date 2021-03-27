python CODE\\run_task.py^
    --task_name CSLinear_Albert_CrossAttn^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 100^
    --evltest_batch_size 12^
    ^
    --cs_num 4^
    --max_qa_len 58^
    --max_cs_len 20^
    --max_seq_len 140^
    --train_batch_size 2^
    --gradient_accumulation_steps 8^
    --learning_rate 2e-5^
    --num_train_epochs 2^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^