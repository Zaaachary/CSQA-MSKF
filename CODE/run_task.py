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
from model.HH_linear import AlbertCrossAttn, BertCrossAttn
from model.AlbertBurger import AlbertBurgerAlpha0, AlbertBurgerAlpha1, AlbertBurgerAlpha2, AlbertBurgerAlpha3, AlbertBurgerAlpha4, AlbertBurgerAlpha5
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
        "Bert_CrossAttn": (BertCrossAttn, ['cs_num', 'max_qa_len', 'max_cs_len']),
        "Albert_Baseline": (AlbertBaseline, []),
        "Albert_AttnMerge": (AlbertAttnMerge, []),
        "Albert_AttnMergeAddTFM": (AlbertAddTFM, []),
        "Albert_CrossAttn": (AlbertCrossAttn, ['cs_num', 'max_qa_len', 'max_cs_len']),
        "Albert_BurgerAlpha0": (AlbertBurgerAlpha0, ['cs_num', 'max_qa_len', 'max_cs_len']),
        "Albert_BurgerAlpha1": (AlbertBurgerAlpha1, ['albert1_layers']),
        "Albert_BurgerAlpha2": (AlbertBurgerAlpha2, ['cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers']),
        "Albert_BurgerAlpha3": (AlbertBurgerAlpha3, ['cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers']),
        "Albert_BurgerAlpha4": (AlbertBurgerAlpha4, ['cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers']),
        "Albert_BurgerAlpha5": (AlbertBurgerAlpha5, ['cs_num', 'max_qa_len', 'max_cs_len', 'albert1_layers'])

    }

    processor_dict = {
        "Origin": Baseline_Processor,
        "OMCS": OMCS_Processor,
        "CSLinear": CSLinear_Processor
    }

    processor_name, model_name = args.task_name.split('_',maxsplit=1)
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
        if 'Origin' not in args.task_name:
            task_str += f'_cs{args.cs_num}'
            task_str += f'_omcsv{args.OMCS_version}'
        
        if 'Burger' in args.task_name:
            task_str += f'_layer{args.albert1_layers}'

        args.result_dir = os.path.join(
            args.result_dir, 
            os.path.basename(args.PTM_model_vocab_dir), 
            args.task_name,
            task_str, ''
            )
    else:
        args.result_dir = args.saved_model_dir
    mkdir_if_notexist(args.result_dir)

    # set logging
    log_file_dir = os.path.join(args.result_dir, 'task_log.txt')
    logging.basicConfig(
        filename = log_file_dir,
        filemode = 'a',
        level = logging.INFO, 
        format = '%(asctime)s %(name)s - %(message)s',
        datefmt = r"%y/%m/%d %H:%M"
        )

    result_dump(args, args.__dict__, 'task_args.json')
    pprint(args.__dict__)

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
        controller.evaluate()
    elif args.mission == 'predict':
        pass

    end = time.time()
    logger.info(f"task total run time {end-start:.2f} second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # other param
    parser.add_argument('--task_name', type=str, help="model & processor will be selected according to task")
    parser.add_argument('--mission', type=str, choices=['train', 'eval', 'predict', 'conti-train'])
    parser.add_argument('--fp16', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_mode', type=str, choices=['epoch', 'step', 'end'], default='end')
    parser.add_argument('--print_step', type=int, default=250)
    parser.add_argument('--eval_after_tacc', type=float, default=0.7)
    parser.add_argument('--evltest_batch_size', type=int, default=8)
    parser.add_argument('--clip_batch_off', action='store_true', default=False, help="clip batch to shortest case")
    
    # task-specific hyper param
    parser.add_argument('--OMCS_version', type=str, default='1')
    parser.add_argument('--albert1_layers', type=int, default=None)
    parser.add_argument('--cs_num', type=int, default=0, help='the cs num of a qc pair')
    parser.add_argument('--max_seq_len', type=int, default=None, help='used where dataprocessor restrain total len')
    parser.add_argument('--max_qa_len', type=int, default=None)
    parser.add_argument('--max_cs_len', type=int, default=None)
    
    # train hyper param
    parser.add_argument('--train_batch_size', type=int, default=4)
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
    --task_name CSLinear_Albert_CrossAttn
    --mission train
    --fp16 0
    --gpu_ids 0
    --save_mode step
    --print_step 100
    --evltest_batch_size 12
    --cs_num 4
    --max_qa_len 56
    --max_cs_len 20
    --max_seq_len 140
    --train_batch_size 2
    --gradient_accumulation_steps 8
    --learning_rate 2e-5
    --num_train_epochs 2
    --warmup_proportion 0.1
    --weight_decay 0.1
    --dataset_dir ..\DATA
    --result_dir  ..\DATA\result
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2
    """

    args = parser.parse_args()
    # args = parser.parse_args(args_str.split())

    main(args)
