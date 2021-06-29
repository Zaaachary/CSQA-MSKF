python CODE/run_csqa_task.py\
    --task_name OMWKCS_MultiSourceFusion\
    --mission predict\
    --fp16 0\
    --seed 42\
    --gpu_ids 5\
    --evltest_batch_size 12\
    --processor_batch_size 24\
    --eval_after_tacc 0.79\
    \
    --without_PTM\
    --model_list Origin WKDT OMCS\
    --max_seq_len 140\
    --max_qa_len 58\
    --WKDT_version 5.0\
    --max_desc_len 45\
    --OMCS_version 3.0\
    --cs_num 3\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --saved_model_dir /data/zhifli/model_save/albert-base-v2/OMWKCS_MultiSourceFusion/30May-2146_seed42_Origin+WKDT+OMCS_62.00%/\
    --encoder_dir_list\
    /data/zhifli/model_save/albert-base-v2/Origin_Albert_Baseline/2030-May17_seed5017_58.31/\
    /data/zhifli/model_save/albert-base-v2/WKDT_Albert_Baseline/0027-May18_seed5017_wkdtv4.0_59.05/\
    /data/zhifli/model_save/albert-base-v2/OMCS_Albert_Baseline/2152-May19_seed42_cs3_omcsv3.0/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2