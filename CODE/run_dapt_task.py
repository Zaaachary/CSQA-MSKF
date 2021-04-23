#! -*- encoding:utf-8 -*-
"""
@File    :   run_dapt_task.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""

import argparse
import logging
import os
import time
from pprint import pprint

from transformers import AlbertTokenizer, BertTokenizer


from dapt_task.data import *
from dapt_task.controller import Dom


logger = logging.getLogger("run_task")
console = logging.StreamHandler();console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s - %(message)s', datefmt = r"%y/%m/%d %H:%M")
console.setFormatter(formatter)
logger.addHandler(console)