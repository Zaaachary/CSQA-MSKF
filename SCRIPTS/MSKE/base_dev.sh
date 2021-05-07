python CODE/run_csqa_task.py\
    --task_name MSKE_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 7\
    --evltest_batch_size 12\
    --knowledge_ensemble \
    \
    --OMCS_version 3.0\
    --WKDT_version 4.0\
    --max_seq_len 130\
    --cs_num 4\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir   /home/zhifli/DATA/model_save/1319-May07_seed42_TMtrain_01_equal_DMtrain_01_equal/\
    --PTM_model_vocab_dir /data/zhifli/transformers-models/albert-base-v2
