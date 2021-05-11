python CODE/run_csqa_task.py\
    --task_name Origin_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 6\
    --evltest_batch_size 16\
    \
    --max_seq_len 128\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /data/zhifli/model_save/albert-xxlarge-v2/Origin_Albert_Baseline/1412-May09_seed509/\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2/
