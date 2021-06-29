#! -*- encoding:utf-8 -*-
"""
@File    :   case_study.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import os
import json
from common import mkdir_if_notexist


def load_csqa(dataset_dir, target):
    raw_csqa = []
    f = open(os.path.join(dataset_dir, 'csqa', f"{target}_rand_split.jsonl"), 'r', encoding='utf-8')
    for line in f:
        raw_csqa.append(json.loads(line.strip()))
    f.close()
    return raw_csqa

def load_dev():
    


if __name__ == "__main__":
    pass