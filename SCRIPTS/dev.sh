python CODE/run_csqa_task.py\
    --task_name OMCS_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 1\
    --evltest_batch_size 12\
    \
    --max_seq_len 80\
    --OMCS_version 3.0\
    --cs_num 2\
    --max_qa_len 0\
    --WKDT_version 0\
    --max_desc_len 0\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1312-Apr26_seed42_cs2_omcsv3.0/\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-xxlarge-v2/
