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

import numpy as np
import torch
from transformers import AlbertTokenizer
from transformers import AlbertConfig
from transformers.optimization import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from model.models import AlbertCSQA
from utils.common import mkdir_if_notexist
from csqa_task import data_processor
from csqa_task.trainer import Trainer
from csqa_task.task import MultipleChoice


def train(train_dataloader, devlp_dataloader, args):
    
    # init
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    model = AlbertCSQA.from_pretrained(args.pretrained_model_dir)
    print(model)

    # train
    trainer = Trainer(
            model, False, device,
            args.print_step, args.output_model_dir, args.fp16)


    t_total = len(train_dataloader) * args.num_train_epochs
    warmup_proportion = args.warmup_proportion

    params = list(model.named_parameters())

    no_decay_keywords = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    def _no_decay(n):
        return any(nd in n for nd in no_decay_keywords)

    parameters = [
        {'params': [p for n, p in params if _no_decay(n)], 'weight_decay': 0.0},
        {'params': [p for n, p in params if not _no_decay(n)],
            'weight_decay': args.weight_decay}
    ]

    optimizer = AdamW(parameters, lr=args.lr, eps=1e-8)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
          optimizer, num_warmup_steps=warmup_proportion * t_total,
          num_training_steps=t_total)

    
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)

    trainer.train(args.num_train_epochs, train_dataloader, devlp_dataloader, save_last=False)



def main(args):
    start = time.time()
    print("start is {}".format(start))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data and preprocess
    tokenizer = AlbertTokenizer.from_pretrained(args.pretrained_vocab_dir)
    print("tokenizer loaded")

    processor = data_processor.CSQAProcessor('DATA', 'train')
    processor.load_data()
    train_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    processor = data_processor.CSQAProcessor('DATA', 'dev')
    processor.load_data()
    deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    # # train
    # train(train_dataloader, deval_dataloader, args)
    task = MultipleChoice(args)
    task.init(AlbertCSQA)
    task.train(train_dataloader, deval_dataloader, save_last=False)

    end = time.time()
    logger.info("start is {}, end is {}".format(start, end))
    logger.info("循环运行时间:%.2f秒"%(end-start))
    with open('./result_1.txt', 'w', encoding='utf-8') as f:
        f.write("循环运行时间:%.2f秒"%(end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 训练过程中的参数
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # 路径参数
    parser.add_argument('--train_file_name', type=str)
    parser.add_argument('--dev_file_name', type=str)
    parser.add_argument('--test_file_name', type=str)
    parser.add_argument('--pred_file_name', type=str)
    parser.add_argument('--output_model_dir', type=str)
    parser.add_argument('--pretrained_model_dir', type=str)
    parser.add_argument('--pretrained_vocab_dir', type=str)

    # 其他参数
    parser.add_argument('--print_step', type=int, default=250)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--fp16', type=int, default=0)

    args = parser.parse_args()
    argstr = """
    --batch_size 2
    --lr 1e-5 
    --num_train_epochs 1 
    --warmup_proportion 0.1 
    --weight_decay 0.1 
    --fp16 0 
    --print_step 100 
    --mission train 
    --train_file_name DATA/csqa/train_data.json 
    --dev_file_name DATA/csqa/dev_data.json 
    --test_file_name DATA/csqa/trial_data.json 
    --pred_file_name  DATA/result/task_result.json 
    --output_model_dir DATA/result/model/
    --pretrained_model_dir DATA/model/albert-large-v2/
    --pretrained_vocab_dir DATA/model/albert-large-v2/
    """
    # args = parser.parse_args(argstr.split())
    print(args)
    main(args)