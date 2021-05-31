python CODE/run_csqa_task.py\
    --task_name OMCS_Albert_Baseline\
    --mission predict\
    --fp16 0\
    --gpu_ids 5\
    --evltest_batch_size 12\
    \
    --max_seq_len 140\
    --OMCS_version 3.0_rank\
    --cs_num 3\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1946-May21_seed42_cs3_omcsv3.0_rank\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-xxlarge-v2
