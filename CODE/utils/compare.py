#! -*- encoding:utf-8 -*-
"""
@File    :   compare.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   

xxlarge
- WKDT  /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1138-Apr23_seed42_wkdtv3.0/   75.59

- Origin /data/zhifli/model_save/albert-xxlarge-v2/Origin_Albert_Baseline/1712-Apr09_seed42/

- OMCS /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1311-Apr26_seed425_cs1_omcsv3.0/
"""
import argparse
import os
import json
from collections import Counter

def load_result(model_dir):
    # file_neme = "right_result.json" "wrong_reuslt.json"
    right_file = os.path.join(model_dir, "right_result.json")
    wrong_file = os.path.join(model_dir, "wrong_result.json")

    f = open(right_file, 'r', encoding='utf-8')
    right_list = json.load(f)
    f.close()

    f = open(wrong_file, 'r', encoding='utf-8')
    wrong_list = json.load(f)
    f.close()

    info = wrong_list.pop(0)

    result = {
        'right_dict': {case['id']:case for case in right_list},
        'wrong_dict': {case['id']:case for case in wrong_list},
        'info' : info
    }        

    return result

def count_vote(result_list):
    csqa_dict = {}

    # origin
    origin = result_list.pop(0)
    origin['right_dict'].update(origin['wrong_dict'])
    for key, value in origin['right_dict'].items():
        csqa_dict[key] = {
            'answerKey': value['answerKey'],
            'predictList': [value['AnswerKey_pred'], ]
        }
    
    # add ohter
    for result in result_list:
        result['right_dict'].update(result['wrong_dict'])

        for key, value in result['right_dict'].items():
            csqa_dict[key]['predictList'].append(value['AnswerKey_pred'])

    # vote
    equal = 0
    for key, value in csqa_dict.items():
        c = Counter(value['predictList'])
        predict, count = c.most_common()[0]
        if count == 1:
            value['predict'] = value['predictList'][0]
            equal += 1
        else:
            value['predict'] = predict

        if predict == value["answerKey"]:
            value['TF'] = "T"
        else:
            value['TF'] = "F"

    # count true
    right, wrong = 0, 0
    for value in csqa_dict.values():
        if value['TF'] == "T":
            right += 1
        elif value['TF'] == "F":
            wrong += 1
    print(f"right[{right}]; wrong[{wrong}]; equal[{equal}]")
    print(f'acc[{right/len(csqa_dict):.6f}]')

def main(args):

    result_list = []
    for file_dir in args.predict_dir:
        result_list.append(load_result(file_dir))

    if args.task_name == 'merge':
        right_dict = {}
        for index, result in enumerate(result_list):
            info = result['info']
            print(f"In result [{index}], right [{info['right']}]; acc [{info['acc']:.7}]")
            right_dict.update(result['right_dict'])

        total_right = len(right_dict)
        acc = total_right / 1221 * 100
        print(f"After merge, right [{total_right}]; acc [{acc:.4f}]")

    if args.task_name == 'vote':
        count_vote(result_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=['compare', 'merge', 'vote'])
    parser.add_argument('--predict_dir', nargs='+')

    args_str = """
    --task_name vote
    --predict_dir 
    /data/zhifli/model_save/albert-xxlarge-v2/Origin_Albert_Baseline/1712-Apr09_seed42/
    /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1311-Apr26_seed425_cs1_omcsv3.0/
    /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1312-Apr26_seed42_cs2_omcsv3.0/
    """
    # /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1138-Apr23_seed42_wkdtv3.0/
    args = parser.parse_args(args_str.split())
    print(args)
    # args = parser.parse_args()

    main(args)
    
