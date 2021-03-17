#! -*- encoding:utf-8 -*-
"""
@File    :   controller.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   任务控制器

- load PTM model or trained model
- train and save model by calling Trainer
- evaluate model by calling Trainer
- make prediction by running model
"""
import logging
logger = logging.getLogger("controller")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    def __init__(self, args):
        self.config = args
        gpu_ids = list(map(int, self.config.gpu_ids.split()))
        self.multi_gpu = (len(gpu_ids) > 1)
        self.device = get_device(gpu_ids)

    def init(self, ModelClass):
        '''
        ModelClass: e.g. modelTC
        '''
        # load model
        if self.config.mission == "train":
            model_dir = self.config.PTM_model_vocab_dir
        else:
            model_dir = self.config.saved_model_dir
        model = ModelClass.from_pretrained(model_dir)
        # print(model)

        if self.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)
            
        self.trainer = Trainer(
            model, self.multi_gpu, self.device,
            self.config.print_step, self.config.result_dir, self.config.fp16)
        self.model = model

    def train(self, train_dataloader, devlp_dataloader):
        # t_total = train_step // args.gradient_accumulation_steps * args.num_train_epochs // device_num
        train_step = len(train_dataloader)
        total_training_step = train_step // self.config.gradient_accumulation_steps * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        # make and set optimizer & scheduler
        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, total_training_step)
        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(
            self.config.num_train_epochs, self.config.gradient_accumulation_steps, train_dataloader, devlp_dataloader, self.config.save_mode)

    def evaluate(self, dataloader):
        record = self.trainer.evaluate(dataloader)
        eval_loss = record[0].avg()
        drn, dan = record.list()[1:]
        logger.info(f"eval: loss {eval_loss:.4f}; acc {int(drn)/int(dan):.4f} ({int(drn)}/{int(dan)})")

    def predict(self, dataloader):
        result = []
        idx = []
        labels = []
        predicts = []

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
