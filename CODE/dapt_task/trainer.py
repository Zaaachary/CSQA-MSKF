#! -*- encoding:utf-8 -*-
"""
@File    :   task_n.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import logging
import pdb

logger = logging.getLogger("trainer")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)

from utils.base_trainer import BaseTrainer
from utils.common import get_device


class Trainer(BaseTrainer):
    def __init__(self, 
        model, multi_gpu, device, 
        print_step, eval_after_tacc,
        output_model_dir, fp16, clip_batch_off, exp_name='csqa'):

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, 
            eval_after_tacc, output_model_dir, v_num=5,
            exp_name=exp_name
        )

        self.fp16 = fp16
        self.clip_batch_off = clip_batch_off
        logger.info(f"fp16: {fp16}; clip_batch_off: {clip_batch_off}")

    def clip_batch(self, batch):
        """
        find the longest seq_len in the batch, and cut all sequence to seq_len
        """
        # print("batch size is {}".format(len(batch[0])))
        input_ids, attention_mask, token_type_ids, sequence_labels, desc_labels = batch

        # [batch_size, 5, max_seq_len]
        batch_size = input_ids.size(0)
        while True:
            # cut seq_len step by step
            end_flag = False
            for i in range(batch_size):
                # traverse batch find if any case has reach the end
                if input_ids[i, -1] != 0:
                    end_flag = True 
            
            if end_flag:
                break
            else:
                input_ids = input_ids[:, :-1]
        
        max_seq_length = input_ids.size(-1)
        attention_mask = attention_mask[:, :max_seq_length]
        token_type_ids = token_type_ids[:, :max_seq_length]
        sequence_labels = sequence_labels[:, :max_seq_length]
        
        # logger.info(f'clip batch to {max_seq_length}')
        
        return input_ids, attention_mask, token_type_ids, sequence_labels, desc_labels
        
    def _forward(self, batch, record):
        if not self.clip_batch_off:
            batch = self.clip_batch(batch)

        batch = tuple(t.to(self.device) for t in batch)
        result = self.model(*batch)  # loss, right_num
        
        # statistic
        result_n = list(map(lambda x:x.item(), result)) # tensor 2 float
        result_n.append(batch[0].shape[0])   # add batch_size
        record.inc(result_n)

        loss = result[0]
        return loss   # loss

    def _report(self, record, mode='Train'):
        '''
        mode: Train, Dev
        '''
        # record: loss, right_num, all_num
        total_loss = record[0].avg()  # utils.common.AvgVar
        masked_lm_loss = record[1].avg()
        right_desc_loss = record[2].avg()

        right_num, all_num = record.list()[-2:]  # right_num, all_num
        output_str = f"{mode}: mlm_loss {masked_lm_loss:.4f}; desc_loss {right_desc_loss:.4f}; desc_acc {int(right_num)/int(all_num):.4f}"

        logger.info(output_str)