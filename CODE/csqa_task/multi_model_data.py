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
from utils.common import get_device, mkdir_if_notexist
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
        self.args = args
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
        for model_index, model_name in enumerate(args.model_list):

            self.make_multisource_dataloader(model_index, model_name, tokenizer, args, shuffle=shuffle)
            self.run_model(model_index, model_name)    # self.model_pooler_batch

        self.models = {}
        torch.cuda.empty_cache()
        
        model_name = self.model_list[0]
        dataloader = self.dataloaders[model_name]
        self.labels = torch.cat([batch[-1] for batch in dataloader], dim=0)

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

    def make_multisource_dataloader(self, model_index, model_name, tokenizer, args, shuffle):
        Processor_dict = {
            'Origin': Baseline_Processor,
            'OMCS': OMCS_Processor,
            'WKDT': Wiktionary_Processor
        }
        if not self.load_cache(model_index, False) or model_index==0:
            logger.info(f"Make dataloader for {model_name}")
            processor = Processor_dict[model_name]
            self.dataloaders[model_name] = processor.make_dataloader(self, tokenizer, args, shuffle=shuffle)

    def run_model(self, model_index, model_name):

        pooler = self.load_cache(model_index)
        if not pooler:
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
            self.save_pooler(batch_pooler, model_index)
            model.to('cpu')
            # torch.cuda.empty_cache()
        else:
            logger.info(f"Load pooler generated from {model_name} model")
            self.model_pooler_batch[model_name] = pooler

    def save_pooler(self, batch_pooler, model_index):
        model_name = self.model_list[model_index]
        save_dir = self.encoder_dir_list[model_index]
        file_name = f"{model_name}_seq{self.args.max_seq_len}_{self.dataset_type}.pt"
        file_dir = os.path.join(save_dir, 'pooler', file_name)
        mkdir_if_notexist(file_dir)
        torch.save(batch_pooler, file_dir)

    def load_cache(self, model_index, return_pooler=True):
        model_name = self.model_list[model_index]
        save_dir = self.encoder_dir_list[model_index]
        file_name = f"{model_name}_seq{self.args.max_seq_len}_{self.dataset_type}.pt"
        file_dir = os.path.join(save_dir, 'pooler', file_name)

        if os.path.isfile(file_dir):
            if return_pooler:
                batch_pooler = torch.load(file_dir)
                return batch_pooler
            else:
                return True
        else:
            return False