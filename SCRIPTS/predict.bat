python CODE\\run_csqa_task.py^
    --task_name Origin_Albert_Baseline^
    --mission eval^
    --fp16 0^
    --gpu_ids 0^
    --save_mode step^
    --print_step 100^
    --eval_after_tacc 0^
    --evltest_batch_size 16^
    ^
    --max_seq_len 54^
    ^
    --dataset_dir DATA^
    --result_dir  DATA/result/^
    --saved_model_dir D:\CODE\Commonsense\CSQA_DATA\model_save\1528-Apr22_seed42^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2^