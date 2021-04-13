python CODE\\run_task.py^
    --task_name CSLinear_Albert_BurgerAlpha6^
    --mission train^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 100^
    --eval_after_tacc 0.58^
    --evltest_batch_size 12^
    --clip_batch_off^
    ^
    --cs_num 4^
    --max_qa_len 58^
    --max_cs_len 20^
    --max_seq_len 140^
    --albert1_layer 10^
    --OMCS_version 1^
    ^
    --train_batch_size 2^
    --gradient_accumulation_steps 4^
    --learning_rate 2e-5^
    --num_train_epochs 2^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^