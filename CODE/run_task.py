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
from transformers import AlbertTokenizer, ElectraTokenizerFast, BertTokenizer

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
        print('tokenizer load error')

def select_task(args):
    if args.task_name == "AlbertAttnMerge":
        return AlbertCSQA, data_processor.Baseline_Processor
    if args.task_name == "AlbertBaseine":
        return AlbertBaseine, data_processor.Baseline_Processor
    elif args.task_name == "AlbertAttnMergeAddTFM":
        return AlbertAddTFM, data_processor.Baseline_Processor
    elif args.task_name == "BertAttRanker":
        return BertAttRanker

def main(args):
    start = time.time()
    set_seed(args)
    print("start in {}".format(start))

    # load data and preprocess
    print("loading tokenizer")
    tokenizer = select_tokenizer(args)
    model, Processor = select_task(args)

    if args.mission == 'train':
        print("loading train set")
        processor = Processor(args.dataset_dir, 'train')
        processor.load_csqa()
        train_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

        print('loading dev set')
        processor = Processor(args.dataset_dir, 'dev')
        processor.load_csqa()
        deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)
    
    else:
        print('loading dev set')
        processor = Processor(args.dataset_dir, 'dev')
        processor.load_csqa()
        deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    # choose model and initalize controller
    controller = MultipleChoice(args)
    controller.init(model)

    # run task accroading to mission
    if args.mission == 'train':
        controller.train(train_dataloader, deval_dataloader, save_last=args.save_last)

    elif args.mission == 'test':
        pass

    end = time.time()
    logger.info("start in {:.0f}, end in {:.0f}".format(start, end))
    logger.info("运行时间:%.2f秒"%(end-start))


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
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # path param
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

    --dataset_dir ../DATA
    --pred_file_dir  ../DATA/result/task_result.json
    --model_save_dir ../DATA/result/TCmodel/
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\albert-base-v2
    """
    args = parser.parse_args(args_str.split())
    # args = parser.parse_args()
    print(args)

    main(args)
