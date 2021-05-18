#! -*- encoding:utf-8 -*-
"""
@File    :   multi_model_data.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import json
from random import random, sample, shuffle
from copy import deepcopy
import logging

logger = logging.getLogger("multi_model_data_processor")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, dataloader

from csqa_task.data import Baseline_Processor, OMCS_Processor, Wiktionary_Processor
from model.Baselines import AlbertBaseline
from utils.common import get_device
from csqa_task.trainer import Trainer


class MultiModel_ProcessorBase(Baseline_Processor, OMCS_Processor, Wiktionary_Processor):
    """
    args
    - model_list: [Origin, OMCS, WKDT]
    - encoder_dir_list: [dir1, dir2, dir3]
    
    """
    def __init__(self, args, dataset_type):
        Baseline_Processor.__init__(self, args, dataset_type)
        OMCS_Processor.__init__(self, args, dataset_type)
        Wiktionary_Processor.__init__(self, args, dataset_type)
        self.batch_size = args.processor_batch_size
        self.model_list = args.model_list
        self.encoder_dir_list = args.encoder_dir_list
        self.models = {}
        self.dataloaders = {}
        self.device = get_device(args.gpu_ids)
        self.model_pooler_batch = {}
        self.labels = None

    def load_data(self):

        logger.info(f"Load [{self.dataset_type}] dataset and knowledge source for {'; '.join(self.model_list)}")
        # load raw data
        self.load_csqa()    # self.raw_csqa
        self.load_omcs()    # self.omcs_cropus
        self.load_wkdt()    # self.wiktionary

        logger.info(f"Make {self.dataset_type} example for {'; '.join(self.model_list)} ")
        # make example
        self.make_csqa()    # self.csqa_examples
        self.inject_omcs()  # self.omcs_examples
        self.inject_wkdt()  # self.wkdt_examples

        self.load_model()

    def load_model(self):
        for model_name, model_dir in zip(self.model_list, self.encoder_dir_list):
            logger.info(f"Load {model_name} model")
            self.models[model_name] =  AlbertBaseline.from_pretrained(model_dir)

    def make_dataloader(self, tokenizer, args, shuffle=True):
        self.make_multisource_dataloader(tokenizer, args)
        self.run_model()    # self.model_pooler_batch
        data = self.make_batch()

        self.batch_size = args.train_batch_size if self.dataset_type in ['train', 'conti-trian'] else args.evltest_batch_size
        
        drop_last=False
        dataset = TensorDataset(*data)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, drop_last=drop_last)

        return dataloader

    def make_batch(self):
        batch = []
        for model_name, model_pooler in self.model_pooler_batch.items():
            batch.append(torch.cat(model_pooler, dim=0))
        batch.append(self.labels)
        return batch

    def make_multisource_dataloader(self, tokenizer, args):
        Processor_dict = {
            'Origin': Baseline_Processor,
            'OMCS': OMCS_Processor,
            'WKDT': Wiktionary_Processor
        }

        for model_name in args.model_list:
            logger.info(f"Make dataloader for {model_name}")
            processor = Processor_dict[model_name]
            self.dataloaders[model_name] = processor.make_dataloader(self, tokenizer, args, shuffle=False)

    def run_model(self):
        model_name = self.model_list[0]
        dataloader = self.dataloaders[model_name]
        self.labels = torch.cat([batch[-1] for batch in dataloader], dim=0)

        for model_name in self.model_list:
            logger.info(f"Run {model_name} model to generate pooler feature")
            dataloader = self.dataloaders[model_name]
            model = self.models[model_name]
            model.to(self.device)
            model.eval()

            batch_pooler = []
            for batch in tqdm(dataloader):
                batch = Trainer.clip_batch(batch)

                batch = list(map(lambda x:x.to(self.device), batch))
                batch = batch[:-1]
                with torch.no_grad():
                    pooler = model._forward(*batch, return_pooler=True)
                    B5, H = pooler.shape
                    pooler = pooler.reshape(-1, 5, H)
                    pooler_cpu = pooler.to('cpu')
                    del pooler
                    batch_pooler.append(pooler_cpu)
            
            self.model_pooler_batch[model_name] = batch_pooler

            del model
            torch.cuda.empty_cache()
        self.models = {}
        torch.cuda.empty_cache()
