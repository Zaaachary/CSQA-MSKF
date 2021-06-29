python CODE/run_csqa_task.py\
    --task_name MSKE_Albert_Baseline\
    --mission eval\
    --fp16 0\
    --gpu_ids 5\
    --evltest_batch_size 12\
    --dev_method top3\
    \
    --OMCS_version 3.0\
    --WKDT_version 4.0\
    --max_seq_len 130\
    --cs_num 8\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir   /data/zhifli/model_save/albert-base-v2/MSKE_Albert_Baseline/0026-May18_seed5017_TMtrian_02_equal_DMtop3_59.13/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2