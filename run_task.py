#! -*- encoding:utf-8 -*-
"""
@File    :   task.py
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

import torch
import numpy as np
from tqdm import tqdm
from transformers import AlbertTokenizer
from transformers.optimization import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from model.models import AlbertCSQA
from model.modelB import AlbertAddTFM
from utils.common import mkdir_if_notexist
from csqa_task import data_processor
from csqa_task.trainer import Trainer
from csqa_task.task import MultipleChoice


def main(args):
    start = time.time()
    print("start in {}".format(start))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data and preprocess
    print("loading tokenizer")
    tokenizer = AlbertTokenizer.from_pretrained(args.pretrained_vocab_dir)

    if args.mission == 'train':
        print("loading train set")
        processor = data_processor.CSQAProcessor('DATA', 'train')
        processor.load_data()
        train_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    print('loading dev set')
    processor = data_processor.CSQAProcessor('DATA', 'dev')
    processor.load_data()
    deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    # run task accroading to mission
    task = MultipleChoice(args)
    if args.mission == 'train':
        # task.init(AlbertCSQA)
        task.init(AlbertAddTFM)
        task.train(train_dataloader, deval_dataloader, save_last=False)
    
    elif args.mission == 'test':
        task.init(AlbertCSQA)
        idx, result, label, predict = task.trial(deval_dataloader)
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

    # 训练过程中的参数
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # 路径参数
    parser.add_argument('--train_file_name', type=str)      # train_data.json
    parser.add_argument('--dev_file_name', type=str)        # dev_data.json
    parser.add_argument('--test_file_name', type=str)       # test_data.json
    parser.add_argument('--pred_file_name', type=str)       # output of predict file
    parser.add_argument('--output_model_dir', type=str)     # 
    parser.add_argument('--pretrained_model_dir', type=str)
    parser.add_argument('--pretrained_vocab_dir', type=str)

    # 其他参数
    parser.add_argument('--print_step', type=int, default=250)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, choices=['train','test'])
    parser.add_argument('--fp16', type=int, default=0)

    args = parser.parse_args()
    print(args)

    main(args)