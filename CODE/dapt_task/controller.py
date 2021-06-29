#! -*- encoding:utf-8 -*-
"""
@File    :   controller.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import json
import logging
import os

from torch.utils.data import dataloader
logger = logging.getLogger("controller")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)

import torch
from tqdm import tqdm
from utils.common import get_device, mkdir_if_notexist, result_dump

from dapt_task.trainer import Trainer


class DomainAdaptivePreTrain:
    
    def __init__(self, args, model_kwargs={}) -> None:
        self.config = args
        self.model_kwargs = model_kwargs    # args for model like cs_num
        
        self.model = None
        self.train_dataloader = None
        self.deval_dataloader = None
        self.test_dataloader = None

        gpu_ids = list(map(int, self.config.gpu_ids.split()))
        self.multi_gpu = (len(gpu_ids) > 1)
        self.device = get_device(gpu_ids)

    def load_model(self, ModelClass):

        if self.config.mission == "train":
            model_dir = self.config.PTM_model_vocab_dir
            model = ModelClass.from_pretrained(model_dir, **self.model_kwargs)
        else:
            model_dir = self.config.saved_model_dir
            if hasattr(ModelClass, 'from_pt'):
                model = ModelClass.from_pt(model_dir, **self.model_kwargs)
            else:
                model = ModelClass.from_pretrained(model_dir, **self.model_kwargs)
            
            if self.multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)

        self.trainer = Trainer(
            model, self.multi_gpu, self.device,
            self.config.print_step, self.config.eval_after_tacc,
            self.config.fp16, self.config.clip_batch_off,
            self.config.nsp,
            self.config.result_dir,
            exp_name=self.config.task_str)
        self.model = model

    def load_data(self, ProcessorClass, tokenizer):
        if self.config.mission in ("train", 'conti-train'):
            # processor = ProcessorClass(self.config, 'dev', tokenizer)
            processor = ProcessorClass(self.config, 'train', tokenizer)
            processor.load_data()
            self.train_dataloader = processor.make_dataloader(shuffle=False)
            # self.train_dataloader = processor.make_dataloader(self.tokenizer, self.config.train_batch_size, False, 128, shuffle=False)
            logger.info("train dataset loaded")

            processor = ProcessorClass(self.config, 'dev', tokenizer)
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(shuffle=False)
            logger.info("dev dataset loaded")
        
        elif self.config.mission == "eval":
            processor = ProcessorClass(self.config, 'dev', tokenizer)
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(shuffle=False)
            self.processor = processor
            logger.info("dev dataset loaded")

        elif self.config.mission == 'predict':
            processor = ProcessorClass(self.config, 'test', tokenizer)
            processor.load_data()
            self.test_dataloader = processor.make_dataloader(shuffle=False)
            self.processor = processor
            logger.info("test dataset loaded")

    def train(self):
        train_dataloader = self.train_dataloader
        deval_dataloader = self.deval_dataloader
        train_step = len(train_dataloader)

        total_training_step = train_step // self.config.gradient_accumulation_steps * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        # make and set optimizer & scheduler
        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.learning_rate)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, total_training_step)
        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        # 断点续训则先进行一次 Eval
        if self.config.mission == 'conti-train':
            right_num = self.evaluate()
            self.trainer.set_best_acc(right_num)
            self.trainer.load_train_info(self.config.saved_model_dir)

        self.trainer.train(
            self.config.num_train_epochs, self.config.gradient_accumulation_steps, train_dataloader, deval_dataloader, self.config.save_mode)

    def evaluate(self):
        pass

    def run_dev(self):
        pass

    def predict_test(self):
        pass
