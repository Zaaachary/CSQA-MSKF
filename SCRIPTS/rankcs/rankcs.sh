python CODE/run_csqa_task.py\
    --task_name RankOMCS_Albert_Baseline\
    --mission rankcs\
    --fp16 0\
    --gpu_ids 6\
    --evltest_batch_size 12\
    --train_batch_size 12\
    \
    --max_seq_len 128\
    --OMCS_version 3.1\
    --cs_num 4\
    \
    --dataset_dir /home/zhifli/DATA/\
    --saved_model_dir /home/zhifli/DATA/model_save/albert-base/1319-May07_seed42_TMtrain_01_equal_DMtrain_01_equal/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2/
