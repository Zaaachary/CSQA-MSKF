#! -*- encoding:utf-8 -*-
"""
@File    :   task_n.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from utils.base_trainer import BaseTrainer
from utils.common import get_device


class Trainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, 
        print_step, output_model_dir, fp16):

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, output_model_dir, v_num=3
        )

        self.fp16 = fp16
        print("fp16 is {}".format(fp16))

    def clip_batch(self, batch):
        """
        find the longest seq_len in the batch, and cut all sequence to seq_len
        """
        # print("batch size is {}".format(len(batch[0])))
        input_ids, attention_mask, token_type_ids, labels = batch
        # [batch_size, 5, max_seq_len]
        batch_size = input_ids.size(0)
        while True:
            # cut seq_len step by step
            end_flag = False
            for i in range(batch_size):
                # traverse batch find if any case has reach the end
                for j in range(5):
                    if input_ids[i, j, -1] != 0:
                        end_flag = True 
            
            if end_flag:
                break
            else:
                input_ids = input_ids[:, :, :-1]
        
        max_seq_length = input_ids.size(2)
        attention_mask = attention_mask[:, :, :max_seq_length]
        token_type_ids = token_type_ids[:, :, :max_seq_length]
        
        return input_ids, attention_mask, token_type_ids, labels
         
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

        # statistic
        result_n = [it.item() for it in result]   # tensor 2 int
        batch_size = batch[0].shape[0]
        result_n.append(batch_size)
        record.inc(result_n)

        return result[0]    # loss

    def _report(self, train_record, devlp_record=None, mode='both'):
        # record: loss, right_num, all_num
        # import pdb; pdb.set_trace()
        train_loss = train_record[0].avg()  # utils.common.AvgVar
        trn, tan = train_record.list()[1:]  # right_num, batch_size
        train_str = f"Train: loss {train_loss:.4f}; acc {int(trn)/int(tan):.4f} ({int(trn)}/{int(tan)})"
        
        if mode == 'both':
            devlp_loss = devlp_record[0].avg()
            drn, dan = devlp_record.list()[1:]  # 335 1221
            devlp_str = f"| Dev: loss {devlp_loss:.4f}; acc {int(trn)/int(tan):.4f} ({int(drn)}/{int(dan)})"
        else:
            devlp_str = ""

        logger.info(f'{train_str} {devlp_str}')
