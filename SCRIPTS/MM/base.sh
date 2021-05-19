python CODE/run_csqa_task.py\
    --task_name OMWKCS_MultiSourceFusion\
    --mission train\
    --fp16 0\
    --seed 42\
    --gpu_ids 0\
    --save_mode step\
    --print_step 100\
    --evltest_batch_size 12\
    --processor_batch_size 24\
    --eval_after_tacc 0.79\
    --clip_batch_off\
    \
    --without_PTM\
    --model_list Origin WKDT OMCS\
    --max_seq_len 140\
    --max_qa_len 58\
    --WKDT_version 4.0\
    --max_desc_len 45\
    --OMCS_version 3.0\
    --cs_num 3\
    --max_cs_len 20\
    \
    --train_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 1e-5\
    --num_train_epochs 8\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    \
    --dataset_dir /home/zhifli/DATA\
    --result_dir  /data/zhifli/model_save\
    --encoder_dir_list\
    /data/zhifli/model_save/albert-base-v2/Origin_Albert_Baseline/2030-May17_seed5017_58.31/\
    /data/zhifli/model_save/albert-base-v2/WKDT_Albert_Baseline/0027-May18_seed5017_wkdtv4.0_59.05/\
    /data/zhifli/model_save/albert-base-v2/MSKE_Albert_Baseline/0026-May18_seed5017_TMtrian_02_equal_DMtop3_59.13/\
    --PTM_model_vocab_dir /home/zhifli/DATA/transformers-models/albert-base-v2