#! -*- encoding:utf-8 -*-
"""
@File    :   base_trainer.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   

https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_with_hard_restarts_schedule_with_warmup
"""

import os
import logging; logger = logging.getLogger("base_trainer")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)

import torch

from tqdm.autonotebook import tqdm
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.optimization import (
    AdamW, get_cosine_with_hard_restarts_schedule_with_warmup)
try:
    from apex import amp
except ImportError:
    print("apex, tensorboard may not imported")
# from torch.utils.tensorboard import SummaryWriter

from .common import Vn, mkdir_if_notexist


class BaseTrainer:
    """
    train & evaluate
    1. self.train(...)
    2. self.evaluate(...)
    3. self.set_optimizer(optimizer)
    4. self.set_scheduler(scheduler)
    5. self.make_optimizer(...)
    6. self.make_scheduler(...)
    7. self.save_model()
    rewrite ↓
    8. self._report()
    9. self._forward()
    """
    def __init__(self, 
        model, multi_gpu, device, 
        print_step, eval_after_tacc, 
        model_save_dir, v_num, exp_name='exp'):
        """
        device: 主device
        multi_gpu: 是否使用了多个gpu
        v_num: 显示的变量数
        """
        self.device = device
        self.multi_gpu = multi_gpu
        self.model = model.to(device)
        self.model_save_dir = model_save_dir
        self.v_num = v_num
        self.train_record = Vn(v_num)
        self.print_step = print_step
        self.eval_after_tacc = eval_after_tacc
        self.best_loss, self.best_acc = float('inf'), 0
        # self.writer = SummaryWriter(f'./DATA/runs/{exp_name}')
        self.start_epoch = -1

    def set_best_acc(self, acc):
        self.best_acc = acc

    def train(self, epoch_num, gradient_accumulation_steps, 
        train_dataloader, dev_dataloader, save_mode='epoch'):
        """
        save_mode: 'step', 'epoch', 'last'
        """
        
        for epoch in range(self.start_epoch + 1, int(epoch_num)):
            self.epoch = epoch
            logger.info(f'---------Epoch: {epoch+1:02}---------')
            self.model.zero_grad()
            self.global_step = 0
            self.train_record.init()

            for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
                self.model.train()
                self._step(batch, gradient_accumulation_steps)

                # step report
                if self.global_step % self.print_step == 0:
                    print(' ')
                    self._report(self.train_record, 'Train')
                    # self.writer.add_scalar(
                    #     'training loss', 
                    #     self.train_record[0].avg(),
                    #     epoch * len(train_dataloader) + self.global_step
                    # )
                    right, all_num = self.train_record.list()[-2:]
                    train_acc = right / all_num
                    self.train_record.init()

                    # do eval only when train_acc greater than eval_after_tacc
                    if save_mode == 'step' and train_acc >= self.eval_after_tacc:
                        dev_record = self.evaluate(dev_dataloader)  # loss, right_num, all_num
                        self._report(dev_record, 'Dev')
                        # self.writer.add_scalar(
                        #     'Develop loss', 
                        #     dev_record[0].avg(),
                        #     epoch * len(train_dataloader) + self.global_step
                        # )
                        dev_list = dev_record.list()
                        cur_loss = dev_list[0]
                        right_num, all_num  = dev_list[-2:]
                        self.save_or_not(cur_loss, right_num)
                        logger.info(f'current best dev acc: [{self.best_acc/all_num:.4f}]')
            else:
                if self.train_record.list()[-1] != 0:
                    self._report(self.train_record)  # last steps not reach print_step

            # epoch report
            dev_record = self.evaluate(dev_dataloader, True)  # loss, right_num, all_num
            self._report(dev_record, 'Dev')
            dev_list = dev_record.list()
            cur_loss = dev_list[0]
            right_num, all_num  = dev_list[-2:]
            if not save_mode == 'last':
                self.save_or_not(cur_loss, right_num)
            logger.info(f'current best dev acc: [{self.best_acc/all_num:.4f}]')
            lr = self.scheduler.get_lr()[1]
            logger.info(f"learning rate: {lr}")

            self.model.zero_grad()

        # end of train
        if save_mode == 'end':
            self.save_model()

    def evaluate(self, dataloader, use_tqdm=False):
        record = Vn(self.v_num)

        if use_tqdm:
            dataloader = tqdm(dataloader)
        else:
            logger.info('evaluating')

        for batch in dataloader:
            
            self.model.eval()
            with torch.no_grad():
                self._forward(batch, record)

        return record

    def _step(self, batch, gradient_accumulation_steps):
        loss = self._forward(batch, self.train_record)

        loss = loss / gradient_accumulation_steps

        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1) 
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        if (self.global_step + 1) % gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

        self.global_step += 1

    def _mean(self, tuples):
        """
        vars 需要是元组
        """
        if self.multi_gpu:
            return tuple(v.mean() for v in tuples)
        return tuples

    def save_or_not(self, loss, acc=0):
        if self.best_acc < acc:
            self.best_acc = acc
            self.best_loss = loss
            self.save_model()
        elif self.best_acc == acc:
            if self.best_loss > loss:
                self.best_loss = loss
                self.save_model()

    def save_model(self):
        mkdir_if_notexist(self.model_save_dir)
        logger.info('save model')
        # self.model.save_pretrained(self.model_save_dir)

        output_model_file = os.path.join(self.model_save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.model_save_dir, CONFIG_NAME)
        output_traininfo_file = os.path.join(self.model_save_dir, "train_info.ckpt")
    
        if hasattr(self.model, "config"):
            self.model.config.to_json_file(output_config_file)
        train_info = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch
        }
        torch.save(self.model.state_dict(), output_model_file)
        torch.save(train_info, output_traininfo_file)

    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            self.model = model
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def make_optimizer(self, weight_decay, lr):
        params = list(self.model.named_parameters())

        no_decay_keywords = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        def _no_decay(n):
            return any(nd in n for nd in no_decay_keywords)

        parameters = [
            {'params': [p for n, p in params if _no_decay(n)], 'weight_decay': 0.0},
            {'params': [p for n, p in params if not _no_decay(n)],
             'weight_decay': weight_decay}
        ]

        optimizer = AdamW(parameters, lr=lr, eps=1e-8)
        return optimizer

    def make_scheduler(self, optimizer, warmup_proportion, t_total):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
          optimizer, num_warmup_steps=warmup_proportion * t_total,
          num_training_steps=t_total)

    def load_train_info(self, model_dir):
        ckpt_dir = os.path.join(model_dir, "train_info.ckpt")
        if os.path.exists(ckpt_dir):
            train_info = torch.load(ckpt_dir)
            self.optimizer.load_state_dict(train_info['optimizer'])
            self.scheduler.load_state_dict(train_info['scheduler'])
            self.start_epoch = train_info['epoch']
        else:
            logger.info(f"{ckpt_dir} not exists")

            lr = self.scheduler.get_lr()[1]
            logger.info(f"current learning rate: {lr}")

    def _forward(self, batch, record):
        """
        rewrite! accroding to actual situation
        """
        batch = tuple(t.to(self.device) for t in batch)
        loss, acc = self.model(*batch)
        loss, acc = self._mean((loss, acc))
        record.inc([loss.item(), acc.item()])
        return loss

    def _report(self, train_record, mode='Train'):
        '''
        rewrite! accroding to actual situation
        '''
        tloss, tacc = train_record.avg()
        print("{mode} loss %.4f acc %.4f" % (tloss, tacc))