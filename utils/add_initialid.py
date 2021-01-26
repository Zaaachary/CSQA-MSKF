#! -*- encoding:utf-8 -*-
"""
@File    :   add_initialid.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import json

base_dir = "DATA/csqa/"

for head in ['dev', 'test', 'train']:
    f = open(base_dir + f"{head}_rand_split.jsonl", 'r', encoding='utf-8')
    examples = []
    for line in f:
        case = json.loads(line.strip())
        examples.append(case)
    f.close()

    f = open(base_dir + f"{head}_rand_split.json", 'w', encoding='utf-8')
    for index, example in enumerate(examples):
        example['idx'] = index
        case_str = json.dumps(example)
        f.write(case_str)
        f.write('\n')
    f.close()
