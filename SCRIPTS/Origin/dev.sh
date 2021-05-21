python CODE/run_csqa_task.py\
    --task_name Origin_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 6\
    --evltest_batch_size 16\
    \
    --max_seq_len 140\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /data/zhifli/model_save/albert-base-v2/Origin_Albert_Baseline/2030-May17_seed5017_58.31/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2/
