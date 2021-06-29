#! -*- encoding:utf-8 -*-
"""
@File    :   task_n.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import logging

from tqdm.autonotebook import tqdm

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
        fp16, clip_batch_off, nsp, 
        output_model_dir, exp_name='csqa'):

        v_num = 5 if nsp else 2

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, 
            eval_after_tacc, output_model_dir, v_num=5,
            exp_name=exp_name
        )

        self.nsp = nsp
        self.fp16 = fp16
        self.clip_batch_off = clip_batch_off
        logger.info(f"fp16: {fp16}; clip_batch_off: {clip_batch_off}")

    def clip_batch(self, batch):
        """
        find the longest seq_len in the batch, and cut all sequence to seq_len
        """
        # print("batch size is {}".format(len(batch[0])))
        if len(batch) == 5:
            input_ids, attention_mask, token_type_ids, sequence_labels, desc_labels = batch
        elif len(batch) == 4:
            input_ids, attention_mask, token_type_ids, sequence_labels = batch

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
        output =  (input_ids, attention_mask, token_type_ids, sequence_labels)
        if len(batch) == 5:
            output += (desc_labels,)
        return output
        
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
        if self.nsp:
            total_loss = record[0].avg()  # utils.common.AvgVar
            masked_lm_loss = record[1].avg()
            right_desc_loss = record[2].avg()

            right_num, all_num = record.list()[-2:]  # right_num, all_num
            output_str = f"{mode}: mlm_loss {masked_lm_loss}; desc_loss {right_desc_loss}; desc_acc {int(right_num)/int(all_num)}"
        else:
            masked_lm_loss = record[0].avg()
            output_str = f"{mode}: mlm_loss {masked_lm_loss}"

        logger.info(output_str)

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

                    if self.nsp:
                        right, all_num = self.train_record.list()[-2:]
                        train_acc = right / all_num
                    self.train_record.init()
                    # do eval only when train_acc greater than eval_after_tacc
                    if save_mode == 'step' and train_acc >= self.eval_after_tacc:
                        dev_record = self.evaluate(dev_dataloader)  # loss, right_num, all_num
                        if self.nsp:
                            dev_list = dev_record.list()
                            cur_loss = dev_list[0]
                            right_num, all_num  = dev_list[-2:]
                            self.save_or_not(cur_loss, right_num)
                            logger.info(f'current best dev acc: [{self.best_acc/all_num:.4f}]')
                        else:
                            mlm_loss = dev_record.list()[0]
                            self.save_or_not(mlm_loss)
                            logger.info(f'current best dev loss: [{self.best_loss}]')

            else:
                self._report(self.train_record)  # last steps not reach print_step

            # epoch report
            dev_record = self.evaluate(dev_dataloader, True)  # loss, right_num, all_num
            self._report(dev_record, 'Dev')
            if self.nsp:
                dev_list = dev_record.list()
                cur_loss = dev_list[0]
                right_num, all_num  = dev_list[-2:]
            
                if not save_mode == 'last':
                    self.save_or_not(cur_loss, right_num)
                logger.info(f'current best dev acc: [{self.best_acc/all_num:.4f}]')
            else:
                mlm_loss = dev_record.list()[0]
                if not save_mode == 'last':
                    self.save_or_not(mlm_loss)
                logger.info(f'current best dev loss: [{self.best_loss}]')

            lr = self.scheduler.get_lr()[1]
            logger.info(f"learning rate: {lr}")

            self.model.zero_grad()

        # end of train
        if save_mode == 'end':
            self.save_model()