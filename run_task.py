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

import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AlbertTokenizer, ElectraTokenizerFast, BertTokenizer

from csqa_task import data_processor
from csqa_task.controller import MultipleChoice
from csqa_task.trainer import Trainer
from model.AttnMerge import AlbertCSQA, AlbertAddTFM
from model.HeadHunter import BertAttRanker
from utils.common import mkdir_if_notexist


def select_tokenizer(args):
    if "albert" in args.pretrained_model_dir:
        return AlbertTokenizer.from_pretrained(args.pretrained_model_dir)
    elif "electra" in args.pretrained_model_dir:
        return ElectraTokenizerFast.from_pretrained(args.pretrained_model_dir)
    elif "bert" in args.pretrained_model_dir:
        return BertTokenizer.from_pretrained(args.pretrained_model_dir)
    else:
        print('tokenizer load error')

def select_task(args):
    if args.task_name == "AlbertAttnMerge":
        return AlbertCSQA, data_processor.Baseline_Processor
    elif args.task_name == "AlbertAttnMergeAddTFM":
        return AlbertAddTFM, data_processor.Baseline_Processor
    elif args.task_name == "BertAttRanker":
        return BertAttRanker
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    start = time.time()
    set_seed(args.seed)
    print("start in {}".format(start))

    # load data and preprocess
    print("loading tokenizer")
    tokenizer = select_tokenizer(args)
    model, Processor = select_task(args)

    if args.mission == 'train':
        print("loading train set")
        processor = Processor('DATA', 'train')
        processor.load_csqa()
        train_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    print('loading dev set')
    processor = Processor('DATA', 'dev')
    processor.load_csqa()
    deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    # choose model and initalize controller
    controller = MultipleChoice(args)
    controller.init(model)

    # run task accroading to mission
    if args.mission == 'train':
        controller.train(train_dataloader, deval_dataloader, save_last=False)

    elif args.mission == 'test':
        idx, result, label, predict = controller.predict(deval_dataloader)
        content = ''
        length = len(result)
        right = 0

        for i, item in enumerate(tqdm(result)):
            if predict[i] == label[i]:
                right += 1
            content += '{},{},{},{},{},{},{},{}\n' .format(idx[i][0], item[0], item[1], item[2], item[3], item[4], label[i], predict[i])

        logger.info("accuracy is {}".format(right/length))
        with open(args.pred_file_name, 'w', encoding='utf-8') as f:
            f.write(content)

    end = time.time()
    logger.info("start in {:.0f}, end in {:.0f}".format(start, end))
    logger.info("运行时间:%.2f秒"%(end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train param
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--fp16', type=int, default=0)

    # path param
    parser.add_argument('--train_file_name', type=str)      # train_data.json
    parser.add_argument('--dev_file_name', type=str)        # dev_data.json
    parser.add_argument('--test_file_name', type=str)       # test_data.json
    parser.add_argument('--pred_file_name', type=str)       # output of predict file
    parser.add_argument('--output_model_dir', type=str)     # 
    parser.add_argument('--pretrained_model_dir', type=str)

    # other param
    parser.add_argument('--print_step', type=int, default=250)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, choices=['train','test'])
    parser.add_argument('--task_name', type=str, default='AlbertAttnMerge')


    # args = parser.parse_args()
    args = parser.parse_args('--batch_size 2 --lr 1e-5 --num_train_epochs 1 --warmup_proportion 0.1 --weight_decay 0.1 --gpu_ids 0 --fp16 0 --print_step 100 --mission train --train_file_name DATA/csqa/train_data.json --dev_file_name DATA/csqa/dev_data.json --test_file_name DATA/csqa/trial_data.json --pred_file_name  DATA/result/task_result.json --output_model_dir DATA/result/model/ --pretrained_model_dir DATA/model/albert-large-v2/'.split())

    print(args)

    main(args)
