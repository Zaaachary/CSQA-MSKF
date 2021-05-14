python CODE/run_csqa_task.py\
    --task_name OMCS_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 6\
    --evltest_batch_size 12\
    \
    --max_seq_len 80\
    --cs_num 2\
    --OMCS_version 3.0\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /home/zhifli/DATA/model_save/albert-xxlarge/OMCS_Albert_Baseline/1312-Apr26_seed42_cs2_omcsv3.0/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-xxlarge-v2/
