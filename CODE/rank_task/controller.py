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
import json
import logging
import os

import torch
from torch.utils.data import dataloader
from tqdm import tqdm
from utils.common import get_device, mkdir_if_notexist, result_dump

from rank_task.trainer import Trainer

logger = logging.getLogger("controller")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(name)s - %(message)s', datefmt=r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)


class TextClassification:
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
            # import pdb; pdb.set_trace()

        if self.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)

        self.trainer = Trainer(
            model, self.multi_gpu, self.device,
            self.config.print_step, self.config.eval_after_tacc,
            self.config.result_dir,
            self.config.fp16, self.config.clip_batch_off,
            exp_name=self.config.task_str)
        self.model = model

    def load_data(self, ProcessorClass, tokenizer):
        self.tokenizer = tokenizer
        if self.config.mission in ("train"):
            processor = ProcessorClass(self.config, 'dev')
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(
                tokenizer, self.config, shuffle=False)
            self.dev_processor = processor
            logger.info("dev dataset loaded")
            
            processor = ProcessorClass(self.config, 'train')
            processor.load_data()
            self.train_dataloader = processor.make_dataloader(
                tokenizer, self.config)
            self.train_processor = processor
            logger.info("train dataset loaded")

        elif self.config.mission == "eval":
            processor = ProcessorClass(self.config, 'dev')
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(
                tokenizer, self.config, shuffle=False)
            self.processor = processor
            logger.info("dev dataset loaded")

        elif self.config.mission == 'predict':
            processor = ProcessorClass(self.config, 'dev')
            processor.load_data()
            self.deval_dataloader = processor.make_dataloader(
                tokenizer, self.config, shuffle=False)
            logger.info("dev dataset loaded")

            processor = ProcessorClass(self.config, 'test')
            processor.load_data()
            self.test_dataloader = processor.make_dataloader(
                tokenizer, self.config, shuffle=False)
            self.processor = processor
            logger.info("test dataset loaded")

    def rankcs(self):
        loss = torch.nn.CrossEntropyLoss(reduction='none')

        for task in ['dev', 'train']:
            loss_list = []
            dataloader = self.deval_dataloader if task == 'dev' else self.train_dataloader
            processor = self.dev_processor if task == 'dev' else self.train_processor

            self.model.eval()
            for batch in tqdm(dataloader):
                if not self.config.clip_batch_off:
                    batch = self.trainer.clip_batch(batch)
                with torch.no_grad():
                    batch = list(map(lambda x: x.to(self.device), batch))
                    labels = batch[-1]
                    batch = batch[:-1]  # rm label

                    logits = self.model.predict(*batch) # [B, 5]
                    batch_loss_list = loss(logits, labels)  # [B]
                    loss_list.extend(batch_loss_list.cpu().tolist())
                    
            csqa_cslist = processor.set_cs_loss(loss_list)
            result_dump(self.config, csqa_cslist, f'{task}_csrank.json', folder='csqa_csrank')


    def train(self):
        train_dataloader = self.train_dataloader
        deval_dataloader = self.deval_dataloader
        train_step = len(train_dataloader)

        total_training_step = train_step // self.config.gradient_accumulation_steps * \
            self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        # make and set optimizer & scheduler
        optimizer = self.trainer.make_optimizer(
            self.config.weight_decay, self.config.learning_rate)
        scheduler = self.trainer.make_scheduler(
            optimizer, warmup_proportion, total_training_step)
        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(
            self.config.num_train_epochs, self.config.gradient_accumulation_steps, train_dataloader, deval_dataloader, self.config.save_mode)

    def evaluate(self):
        dataloader = self.deval_dataloader
        record = self.trainer.evaluate(dataloader, True)
        eval_loss = record[0].avg()
        drn, dan = record.list()[1:]
        logger.info(
            f"eval: loss {eval_loss:.4f}; acc {int(drn)/int(dan):.4f} ({int(drn)}/{int(dan)})")
        return drn

    def run_dev(self, file_prefix=''):
        '''
        run model to predict dev set, and set right/wrong result to file.
        '''
        logits_list, predict_list = [], []
        dataloader = self.deval_dataloader
        self.model.eval()

        for batch in tqdm(dataloader):
            batch = batch[:-1]  # rm label
            with torch.no_grad():
                batch = list(map(lambda x: x.to(self.device), batch))
                logits = self.model.predict(*batch)
                logits_list.extend(logits.cpu().numpy().tolist())
                predict_list.extend(torch.argmax(
                    logits, dim=1).cpu().numpy().tolist())

        csqa_dev = self.processor.make_dev(predict_list, logits_list)

        right, wrong = [], []
        for case in csqa_dev:
            if case['AnswerKey_pred'] == case['answerKey']:
                right.append(case)
            else:
                wrong.append(case)

        summary = {'total': len(csqa_dev), 'right': len(right), 'wrong': len(
            wrong), 'acc': str(len(right)/len(csqa_dev)*100)+'%'}
        wrong.insert(0, summary)
        result_dump(self.config, right, file_prefix +
                    'right_result.json', folder='dev_result')
        result_dump(self.config, wrong, file_prefix +
                    'wrong_result.json', folder='dev_result')

        logger.info(f"eval: acc {len(right)}/{len(csqa_dev)}={summary['acc']}")

    def run_knowledge_ensemble_dev(self):
        ke_method_list = self.processor.ke_method_list
        for index, method in enumerate(ke_method_list):
            logger.info(f'dev in {method}')
            self.processor.remake_data(method)
            self.deval_dataloader = self.processor.make_dataloader(
                self.tokenizer, self.config, shuffle=False)

            self.run_dev(str(index) + method + '_')

    def predict_test(self, file_prefix='', evaluate=True):
        if evaluate:
            self.evaluate()
            
        predict_list = []
        dataloader = self.test_dataloader
        self.model.eval()
        for batch in tqdm(dataloader):
            if not self.config.clip_batch_off:
                batch = self.trainer.clip_batch(batch)
            batch = batch[:-1]  # rm label
            with torch.no_grad():
                batch = list(map(lambda x: x.to(self.device), batch))
                logits = self.model.predict(*batch)
                predict_list.extend(torch.argmax(
                    logits, dim=1).cpu().numpy().tolist())

        raw_csqa = self.processor.set_predict_labels(predict_list)
        result_dump(self.config, raw_csqa, file_prefix + 'predict.json', folder='predict_result')

    def predict_knowledge_ensemble_test(self):
        ke_method_list = self.processor.ke_method_list
        for index, method in enumerate(ke_method_list):
            logger.info(f'predict test in {method}')
            self.processor.remake_data(method)
            self.test_dataloader = self.processor.make_dataloader(
                self.tokenizer, self.config, shuffle=False)

            self.predict_test(str(index) + method + '_', evaluate=False)