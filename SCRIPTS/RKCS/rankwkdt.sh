python CODE/run_csqa_task.py\
    --task_name RankWKDT_Albert_Baseline\
    --mission rankcs\
    --fp16 0\
    --gpu_ids 7\
    --evltest_batch_size 12\
    --train_batch_size 12\
    \
    --max_seq_len 130\
    --max_qa_len 54\
    --max_desc_len 40\
    --WKDT_version 5.0\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /data/zhifli/model_save/albert-base-v2/WKDT_Albert_Baseline/0027-May18_seed5017_wkdtv4.0_59.05/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2
