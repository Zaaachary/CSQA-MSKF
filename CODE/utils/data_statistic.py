#! -*- encoding:utf-8 -*-
"""
@File    :   data_statistic.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import json
import random

# omcs_dev = r"D:\CODE\Commonsense\CSQA_DATA\omcs\omcs_v3.0_15\dev_rand_split_omcs.json"
wkdt_dev = r"D:\CODE\Commonsense\CSQA_DATA\wkdt\wiktionary_v5_rank\dev_concept.json"

f = open(wkdt_dev, 'r', encoding='utf-8')
examples = json.load(f)
f.close()

test_examples = random.sample(examples, k=10)
total_count = 0
noisy_count = 0

for index, example in enumerate(test_examples):
    print(f"case [{index}]")
    print(example['id'])
    print(example['question'])
    print(example['question_concept'])
    # print(example['query'])
    print(example['choice'])
    print(len(example['cs_list']))
    for cs in example['cs_list']:
        print(cs)
    bad = int(input('how many bad?'))
    noisy_count += bad
    total_count += len(example['cs_list'])

print("total_count:", total_count)
print("noisy_count:", noisy_count)
print(noisy_count/total_count)
