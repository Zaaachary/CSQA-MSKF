#! -*- encoding:utf-8 -*-
"""
@File    :   run_task.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import argparse
import logging
import os
import time
from pprint import pprint

from transformers import AlbertTokenizer, BertTokenizer

from csqa_task.data import *
from csqa_task.controller import MultipleChoice

from model.AttnMerge import AlbertAddTFM, AlbertAttnMerge
from model.Baselines import AlbertBaseline, BertBaseline
from model.HH_linear import AlbertCrossAttn
from model.AlbertBurger import AlbertBurgerAlpha0, AlbertBurgerAlpha2,  AlbertBurgerAlpha6

from model.HeadHunter import AlbertAttRanker

from utils.common import mkdir_if_notexist, result_dump, set_seed

logger = logging.getLogger("run_task")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)


def select_tokenizer(args):
    if "albert" in args.PTM_model_vocab_dir:
        return AlbertTokenizer.from_pretrained(args.PTM_model_vocab_dir)
    elif "bert" in args.PTM_model_vocab_dir:
        return BertTokenizer.from_pretrained(args.PTM_model_vocab_dir)
    else:
        logger.error("No Tokenizer Matched")

def select_task(args):
    '''
    task_name format: [processor_name]_[model name]
    '''
    model_dict = {
        "Bert_Baseline": (BertBaseline, []),

        "Albert_Baseline": (AlbertBaseline, []),
        "Albert_AttnMerge": (AlbertAttnMerge, []),
        "Albert_AttnMergeAddTFM": (AlbertAddTFM, []),

        "Albert_AttnRanker": (AlbertAttRanker, ['cs_num',]),

        "Albert_CrossAttn": (AlbertCrossAttn, ['cs_num', 'max_qa_len', 'max_cs_len']),

        "Albert_BurgerAlpha0": (AlbertBurgerAlpha0, ['model_cs_num', 'max_qa_len', 'max_cs_len']),
        # "Albert_BurgerAlpha1": (AlbertBurgerAlpha1, ['albert1_layers']),
        "Albert_BurgerAlpha2": (AlbertBurgerAlpha2, ['model_cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers']),
        # 3 4 5
        # "Albert_BurgerAlphaX": (AlbertBurgerAlphaX, ['cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers']),
        "Albert_BurgerAlpha6": (AlbertBurgerAlpha6, ['model_cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers'])
    }

    processor_dict = {
        "Origin": Baseline_Processor,
        "OMCS": OMCS_Processor,
        "WKDT": Wiktionary_Processor,
        "MSKE": MSKE_Processor,
        "OMCSrerank": OMCS_rerank_Processor,
        "CSLinear": CSLinear_Processor,
        "CSLE": CSLinearEnhanced_Processor,
    }

    processor_name, model_name = args.task_name.split('_', maxsplit=1)
    ModelClass, args_list = model_dict[model_name]
    ProcessorClass = processor_dict[processor_name]

    model_kwargs = {arg: args.__dict__[arg] for arg in args_list}

    return ModelClass, ProcessorClass, model_kwargs

def set_result(args):
    '''
    set result dir name accroding to the task
    '''
    if args.mission in ('train', 'conti-train'):
        task_str = time.strftime(r'%H%M-%b%d') + f'_seed{args.seed}'
        if 'OMCS' in args.task_name or 'CSLinear' in args.task_name:
            task_str += f'_cs{args.cs_num}'
            task_str += f'_omcsv{args.OMCS_version}'
        
        if 'Burger' in args.task_name:
            task_str += f'_layer{args.albert1_layers}'

        if 'WKDT' in args.task_name:
            task_str += f'_wkdtv{args.WKDT_version}'

        if 'MSKE' in args.task_name:
            task_str += f'_TM{args.train_method}_DM{args.dev_method}'

        args.result_dir = os.path.join(
            args.result_dir, 
            os.path.basename(args.PTM_model_vocab_dir), 
            args.task_name,
            task_str, ''
            )
        args.task_str = task_str

        log_file_dir = os.path.join(args.result_dir, 'train_task_log.txt')
        args_file_name = "train_task_args.json"

    else:
        args.task_str = 'predict or dev'
        args.result_dir = args.saved_model_dir
        log_file_dir = os.path.join(args.result_dir, 'test_eval_task_log.txt')
        args_file_name = "test_eval_task_args.json"

    mkdir_if_notexist(args.result_dir)

    result_dump(args, args.__dict__, args_file_name)
    pprint(args.__dict__)

    # set logging
    logging.basicConfig(
        filename = log_file_dir,
        filemode = 'a',
        level = logging.INFO, 
        format = '%(asctime)s %(name)s - %(message)s',
        datefmt = r"%y/%m/%d %H:%M"
        )

def main(args):
    start = time.time()
    logger.info(f"start in {start}")
    set_result(args)
    set_seed(args)

    # load data and preprocess
    logger.info(f"select tokenizer and model for task {args.task_name}")
    tokenizer = select_tokenizer(args)
    model, Processor, model_kwargs = select_task(args)

    # initalize controller by model
    controller = MultipleChoice(args, model_kwargs)
    controller.load_model(model)
    controller.load_data(Processor, tokenizer)

    # run task accroading to mission
    if args.mission in ('train', 'conti-train'):
        controller.train()
    elif args.mission == 'eval':
        if args.knowledge_ensemble:
            controller.run_knowledge_ensemble_dev()
        else:
            controller.run_dev()
    elif args.mission == 'predict':
        if args.knowledge_ensemble:
            controller.predict_knowledge_ensemble_test()
        else:
            controller.predict_test()

    end = time.time()
    logger.info(f"task total run time {end-start:.2f} second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device param
    parser.add_argument('--task_name', type=str, help="model & processor will be selected according to task")
    parser.add_argument('--mission', type=str, choices=['train', 'eval', 'predict', 'conti-train'])
    parser.add_argument('--fp16', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip_batch_off', action='store_true', default=False, help="clip batch to shortest case")
    
    # dev param
    parser.add_argument('--save_mode', type=str, choices=['epoch', 'step', 'end'], default=None)
    parser.add_argument('--print_step', type=int, default=None)
    parser.add_argument('--eval_after_tacc', type=float, default=0)
    parser.add_argument('--evltest_batch_size', type=int)
    parser.add_argument('--dev_method', type=str, default=None)
    parser.add_argument('--knowledge_ensemble', action='store_true')
    
    # task-specific hyper param
    parser.add_argument('--max_seq_len', type=int, default=None, help='used where dataprocessor restrain total len')
    parser.add_argument('--max_qa_len', type=int, default=None)
    parser.add_argument('--max_cs_len', type=int, default=None)
    parser.add_argument('--max_desc_len', type=int, default=None)
    parser.add_argument('--cs_num', type=int, default=0, help='the cs num of a qc pair')
    parser.add_argument('--model_cs_num', type=int, default=0, help='the cs num of a qc pair')
    parser.add_argument('--train_method', type=str, default=None)
    parser.add_argument('--OMCS_version', type=str, default=None)
    parser.add_argument('--WKDT_version', type=str, default=None)
    parser.add_argument('--albert1_layers', type=int, default=None)

    # train hyper param
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # data param
    parser.add_argument('--dataset_dir', type=str, default='../DATA')
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--saved_model_dir', type=str, default=None)
    parser.add_argument('--PTM_model_vocab_dir', type=str, default=None)

    args_str = r"""
    --task_name MSKE_Albert_Baseline
    --mission eval
    --fp16 0
    --gpu_ids 0
    --evltest_batch_size 12
    --knowledge_ensemble 
    --OMCS_version 3.0
    --WKDT_version 4.0
    --max_seq_len 130
    --cs_num 4
    --dataset_dir D:\CODE\Commonsense\CSQA_DATA
    --saved_model_dir   D:\CODE\Commonsense\CSQA_DATA\model_save\1319-May07_seed42_TMtrain_01_equal_DMtrain_01_equal
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2
    """

    args = parser.parse_args()
    # args = parser.parse_args(args_str.split())

    main(args)
