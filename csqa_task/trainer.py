#! -*- encoding:utf-8 -*-
"""
@File    :   task_n.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch

from utils.base_trainer import BaseTrainer
from utils.common import get_device

class Trainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, 
        print_step, output_model_dir, fp16):

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, output_model_dir, vn=3
        )

        self.fp16 = fp16
        print("fp16 is {}".format(fp16))

    def clip_batch(self, batch):
        """
        find the longest seq_len in the batch, and cut all sequence to seq_len
        """
        # print("batch size is {}".format(len(batch[0])))
        idx, input_ids, attention_mask, token_type_ids, labels = batch
        # [batch_size, 5, max_seq_len]
        batch_size = input_ids.size(0)
        while True:
            # cut seq_len step by step
            end_flag = False
            for i in range(batch_size):
                # traverse batch find if any case has reach the end
                if input_ids[i, 0, -1] != 0:
                    end_flag = True
                if input_ids[i, 1, -1] != 0:
                    end_flag = True 
            
            if end_flag:
                break
            else:
                input_ids = input_ids[:, :, :-1]
        
        max_seq_length = input_ids.size(2)
        attention_mask = attention_mask[:, :, :max_seq_length]
        token_type_ids = token_type_ids[:, :, :max_seq_length]
        
        return idx, input_ids, attention_mask, token_type_ids, labels
        
    def _step(self, batch):
        loss = self._forward(batch, self.train_record)
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1) 
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.global_step += 1
        
    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            
            self.model = model
        self.optimizer = optimizer

    def _forward(self, batch, record):
        batch = self.clip_batch(batch)
        batch = tuple(t.to(self.device) for t in batch)
        result = self.model(*batch)
        result = self._mean(result)
        # record
        import pdb; pdb.set_trace()
        record.inc([it.item() for it in result])
        return result[0]

    def _report(self, train_record, devlp_record):
        # record: loss, right_num, all_num
        # import pdb; pdb.set_trace()
        train_loss = train_record[0].avg()  # utils.common.AvgVar
        devlp_loss = devlp_record[0].avg()

        trn, tan = train_record.list()[1:]  # 76, 400  Vn -> list  [29.43147110939026, 6, 40.0]
        drn, dan = devlp_record.list()[1:]  # 335 1221

        logger.info(f'\n____Train: loss {train_loss:.4f} {int(trn)}/{int(tan)} = {int(trn)/int(tan):.4f} |'
              f' Devlp: loss {devlp_loss:.4f} {int(drn)}/{int(dan)} = {int(drn)/int(dan):.4f}')
