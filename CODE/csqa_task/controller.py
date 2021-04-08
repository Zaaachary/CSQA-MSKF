#! -*- encoding:utf-8 -*-
"""
@File    :   controller.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   任务控制器

- load PTM model or trained model
- load data by calling Processor
- train and save model by calling Trainer
- evaluate model by calling Trainer
- make prediction by running model
"""
import logging

from torch.utils.data import dataloader
logger = logging.getLogger("controller")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)

import torch
from tqdm import tqdm
from utils.common import get_device, mkdir_if_notexist

from csqa_task.trainer import Trainer


class MultipleChoice:
    """
    1. self.init()
    2. self.train(...)
    3. cls.load(...)
    """
    def __init__(self, args, model_kwargs={}):
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
        '''
        ModelClass: e.g. modelTC
        '''
        # load model
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
            self.config.result_dir,
            self.config.fp16, self.config.clip_batch_off)
        self.model = model

    def load_data(self, ProcessorClass, tokenizer):
        if self.config.mission in ("train", 'conti-train'):
            processor = ProcessorClass(self.config, 'train')
            processor.load_data()
            self.train_dataloader = processor.make_dataloader(
                tokenizer, self.config)
            # self.train_dataloader = processor.make_dataloader(self.tokenizer, self.config.train_batch_size, False, 128, shuffle=False)
            logger.info("train dataset loaded")

            processor = ProcessorClass(self.config, 'dev')
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(
                tokenizer, self.config)
            logger.info("dev dataset loaded")
        
        elif self.config.mission == "eval":
            processor = ProcessorClass(self.config, 'dev')
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(
                tokenizer, self.config)
            logger.info("dev dataset loaded")

        elif self.config.mission == 'predict':
            processor = ProcessorClass(self.config, 'test')
            processor.load_data()
            self.test_dataloader = processor.make_dataloader(
                tokenizer, self.config, shuffle=False)
            logger.info("test dataset loaded")

    def train(self):
        # t_total = train_step // args.gradient_accumulation_steps * args.num_train_epochs // device_num
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
            self.evaluate()

        self.trainer.train(
            self.config.num_train_epochs, self.config.gradient_accumulation_steps, train_dataloader, deval_dataloader, self.config.save_mode)

    def evaluate(self):
        dataloader = self.deval_dataloader
        record = self.trainer.evaluate(dataloader, True)
        eval_loss = record[0].avg()
        drn, dan = record.list()[1:]
        logger.info(f"eval: loss {eval_loss:.4f}; acc {int(drn)/int(dan):.4f} ({int(drn)}/{int(dan)})")

    def predict(self):
        result = []
        idx = []
        labels = []
        predicts = []

        dataloader = self.deval_dataloader

        for batch in tqdm(dataloader):
            self.model.eval()
            with torch.no_grad():
                ret = self.model.predict(batch[0].to(self.device),batch[1].to(self.device),batch[2].to(self.device),batch[3].to(self.device))
                idx.extend(batch[0].cpu().numpy().tolist())
                result.extend(ret.cpu().numpy().tolist())
                labels.extend(batch[4].numpy().tolist())
                predicts.extend(torch.argmax(ret, dim=1).cpu().numpy().tolist())

        return idx, result, labels, predicts

    @classmethod
    def load(cls, config, ConfigClass, ModelClass):
        gpu_ids = list(map(int, config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
        device = get_device(gpu_ids)

        srt = cls(config)
        srt.device = device
        srt.trainer = Trainer.load_model(
            ConfigClass, ModelClass, multi_gpu, device,
            config.print_step, config.output_model_dir, config.fp16)

        return srt
