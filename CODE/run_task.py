#! -*- encoding:utf-8 -*-
"""
@File    :   run_task.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import argparse
import random
import time

import logging; logging.getLogger("run_task").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tqdm import tqdm
from transformers import AlbertTokenizer, BertTokenizer

from csqa_task import data_processor
from csqa_task.controller import MultipleChoice
from csqa_task.trainer import Trainer
from utils.common import mkdir_if_notexist, set_seed

from model.AttnMerge import AlbertCSQA, AlbertAddTFM
from model.Baselines import AlbertBaseine
from model.HeadHunter import BertAttRanker


def select_tokenizer(args):
    if "albert" in args.PTM_model_vocab_dir:
        return AlbertTokenizer.from_pretrained(args.PTM_model_vocab_dir)
    elif "bert" in args.PTM_model_vocab_dir:
        return BertTokenizer.from_pretrained(args.PTM_model_vocab_dir)
    else:
        logger.error("No Tokenizer Matched")

def select_task(args):
    if args.task_name == "AlbertAttnMerge":
        return AlbertCSQA, data_processor.Baseline_Processor
    if args.task_name == "AlbertBaseine":
        return AlbertBaseine, data_processor.Baseline_Processor
    elif args.task_name == "AlbertAttnMergeAddTFM":
        return AlbertAddTFM, data_processor.Baseline_Processor
    elif args.task_name == "Bert_OMCS_AttRanker":
        return BertAttRanker, data_processor.OMCS_Processor
    elif args.task_name == "Albert_OMCS_Baseline":
        return AlbertBaseine, data_processor.OMCS_Processor
        

def main(args):
    start = time.time()
    set_seed(args)
    print("start in {}".format(start))

    # load data and preprocess
    logger.info(f"select tokenizer and model for task {args.task_name}")
    tokenizer = select_tokenizer(args)
    model, Processor = select_task(args)

    if args.mission == 'train':
        processor = Processor(args, 'train')
        processor.load_data()
        train_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)
        logger.info("train dataset loaded")

        processor = Processor(args, 'dev')
        processor.load_data()
        deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)
        logger.info("dev dataset loaded")

    elif args.mission == 'eval':
        processor = Processor(args, 'dev')
        processor.load_data()
        deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)
        logger.info("dev dataset loaded")

    elif args.mission == 'predict':
        processor = Processor(args, 'test')
        processor.load_data()
        deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)
        logger.info("test dataset loaded")

    # initalize controller by model
    controller = MultipleChoice(args)
    controller.init(model)

    # run task accroading to mission
    if args.mission == 'train':
        controller.train(train_dataloader, deval_dataloader, save_last=args.save_last)

    elif args.mission == 'eval':
        pass

    elif args.mission == 'predict':
        pass

    end = time.time()
    logger.info("task start in {:.0f}, end in {:.0f}".format(start, end))
    logger.info("total run time:%.2f second"%(end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # other param
    parser.add_argument('--task_name', type=str, default='AlbertAttnMerge')
    parser.add_argument('--mission', type=str, choices=['train','eval', 'predict'])
    parser.add_argument('--fp16', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_last', action="store_true")
    parser.add_argument('--print_step', type=int, default=250)
    
    # hyper param
    parser.add_argument('--cs_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # data param
    parser.add_argument('--dataset_dir', type=str, default='../DATA')
    parser.add_argument('--pred_file_dir', type=str)       # output of predict file
    parser.add_argument('--model_save_dir', type=str, default=None)     # 
    parser.add_argument('--PTM_model_vocab_dir', type=str, default=None)

    args_str = r"""
    --task_name AlbertBaseine
    --mission train
    --fp16 0
    --gpu_ids -1
    --print_step 100

    --batch_size 4
    --lr 1e-5
    --num_train_epochs 4
    --warmup_proportion 0.1
    --weight_decay 0.1
    --cs_num 2

    --dataset_dir ../DATA
    --pred_file_dir  ../DATA/result/task_result.json
    --model_save_dir ../DATA/result/TCmodel/
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2
    """
    # args = parser.parse_args(args_str.split())
    args = parser.parse_args()
    print(args)

    main(args)
