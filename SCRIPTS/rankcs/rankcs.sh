python CODE/run_csqa_task.py\
    --task_name RankOMCS_Albert_Baseline\
    --mission rankcs\
    --fp16 0\
    --gpu_ids 5\
    --evltest_batch_size 12\
    --train_batch_size 12\
    \
    --max_seq_len 128\
    --OMCS_version 3.0\
    --cs_num 8\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /data/zhifli/model_save/albert-base-v2/OMCS_Albert_Baseline/2152-May19_seed42_cs3_omcsv3.0/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2
