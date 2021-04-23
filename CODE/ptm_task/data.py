#! -*- encoding:utf-8 -*-
"""
@File    :   data.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import json
from random import sample

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from ptm_task.example import *


class ProcessorBase(object):

    def __init__(self, args, dataset_type) -> None:
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = dataset_type

    def load_data(self):
        pass

    