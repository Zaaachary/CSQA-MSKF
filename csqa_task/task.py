#! -*- encoding:utf-8 -*-
"""
@File    :   task.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import torch
from tqdm import tqdm

from utils.common import get_device
from csqa_task.trainer import Trainer

class MultipleChoice:
    """
    1. self.init()
    2. self.train(...)
    3. cls.load(...)
    """
    def __init__(self, args):
        self.config = args

    def init(self, ModelClass):
        gpu_ids = list(map(int, self.config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
#        multi_gpu = gpu_ids
        self.device = get_device(gpu_ids)

        print('init_model', self.config.pretrained_model_dir)
        model = ModelClass.from_pretrained(self.config.pretrained_model_dir)
        print(model)

        if multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        self.trainer = Trainer(
            model, multi_gpu, self.device,
            self.config.print_step, self.config.output_model_dir, self.config.fp16)
        self.model = model

    def train(self, train_dataloader, devlp_dataloader, save_last=True):
        t_total = len(train_dataloader) * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, t_total)

        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(
            self.config.num_train_epochs, train_dataloader, devlp_dataloader, save_last=save_last)

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

    def trial(self, dataloader):
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
