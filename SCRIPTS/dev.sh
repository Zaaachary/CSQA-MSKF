python CODE/run_csqa_task.py\
    --task_name WKDT_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 5\
    --evltest_batch_size 12\
    \
    --max_seq_len 130\
    --WKDT_version 5.0\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /home/zhifli/DATA/model_save/WKDT_Albert_Baseline/1829-May04_seed5004_wkdtv4.0/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-xxlarge-v2/
