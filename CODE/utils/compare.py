#! -*- encoding:utf-8 -*-
"""
@File    :   compare.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   

xxlarge
- WKDT  /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1138-Apr23_seed42_wkdtv3.0/   75.59
- WKDT  /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1829-May04_seed5004_wkdtv4.0/   79.93

- Origin /data/zhifli/model_save/albert-xxlarge-v2/Origin_Albert_Baseline/1712-Apr09_seed42/

- OMCS /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1311-Apr26_seed425_cs1_omcsv3.0/
"""
import argparse
import os
import json
from collections import Counter
import pdb


def load_result(args, model_dir, method=None):
    if args.dev:
        if not method:
            right_file = os.path.join(model_dir, "right_result.json")
            wrong_file = os.path.join(model_dir, "wrong_result.json")
        else:
            right_file = os.path.join(model_dir, f"{method}_right_result.json")
            wrong_file = os.path.join(model_dir, f"{method}_wrong_result.json")

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
    else:
        if not method:
            predict_file = os.path.join(model_dir, "predict.json")
        else:
            predict_file = os.path.join(model_dir, f"{method}_predict.json")
        
        f = open(predict_file, 'r', encoding='utf-8')
        result = json.load(f)
        f.close()
        result = {case['id']: case for case in result}

    return result

def count_vote(args, result_list, choose_top, task_name):
    csqa_dict = {}

    # origin
    top_result_index = 0
    if choose_top:
        top_acc = 0
        for index, result in enumerate(result_list):
            acc = result[0]['info']['right'] / result[0]['info']['total']
            if acc > top_acc:
                top_acc = acc
                top_result_index = index
        print(f'dev top is {result_list[top_result_index][1]}')

    # import pdb; pdb.set_trace()
    origin = result_list.pop(top_result_index)
    if args.dev:
        origin = origin[0]
        origin['right_dict'].update(origin['wrong_dict'])
    
        for key, value in origin['right_dict'].items():
            csqa_dict[key] = {
                'answerKey': value['answerKey'],
                'predictList': [value['AnswerKey_pred'], ],
                'logitsList': [[choice['logit'] for choice in value['choices']], ]
            }

            # add ohter
        for result in result_list:
            result = result[0]
            result['right_dict'].update(result['wrong_dict'])

            for key, value in result['right_dict'].items():
                csqa_dict[key]['predictList'].append(value['AnswerKey_pred'])
                csqa_dict[key]['logitsList'].append([choice['logit'] for choice in value['choices']])

    else:
        origin = origin[0]
        for key, value in origin.items():
            csqa_dict[key] = {
                'predictList': [value['AnswerKey_pred'], ],
                # 'logitsList': [[choice['logit'] for choice in value['choices']], ]
            }
        
        for result in result_list:
            result = result[0]

            for key, value in result.items():
                csqa_dict[key]['predictList'].append(value['AnswerKey_pred'])
                # csqa_dict[key]['logitsList'].append([choice['logit'] for choice in value['choices']])
        
    # vote
    equal = 0
    for key, value in csqa_dict.items():
        equal_flag = False
        if task_name in ['vote', 'compare']:
            c = Counter(value['predictList'])
            common_list = list(c.most_common())
            predict = common_list[0][0]
            if len(common_list) > 1:
                if common_list[0][1] == common_list[1][1]:
                    equal_flag = True

            if equal_flag:
                value['predict'] = value['predictList'][0]
                equal += 1
            else:
                value['predict'] = predict

        elif task_name == 'vote_logit':
            max_logit = [0, 0, 0, 0, 0]
            for logits in value['logitsList']:
                for index, choice in enumerate(logits):
                    max_logit[index] = max(max_logit[index], choice)

            max_index, max_value = 0, 0
            for index, logit in enumerate(max_logit):
                if logit > max_value:
                    max_value = logit
                    max_index = index

            value['predict'] = chr(ord('A') + max_index)

            # import pdb; pdb.set_trace()

        if args.dev:
            if value['predict'] == value["answerKey"]:
                value['TF'] = "T"
            else:
                value['TF'] = "F"

    if args.dev:
        # count true
        right, wrong = 0, 0
        for value in csqa_dict.values():
            if value['TF'] == "T":
                right += 1
            elif value['TF'] == "F":
                wrong += 1
        print(f"right[{right}]; wrong[{wrong}]; equal[{equal}]")
        print(f'acc[{right/len(csqa_dict):.6f}]')
    else:
        print(f"equal[{equal}]")
    
    return csqa_dict
        
def compare(args, csqa_dict, result_list):
    count_list = [0 for _ in range(len(result_list))]
    for index, result in enumerate(result_list):

        for key, value in result[0].items():
            predict = csqa_dict[key]['predict']
            if value['AnswerKey_pred'] == predict:
                count_list[index] += 1
        else:
            print(f"{result[1]} : [{count_list[index]}]")
    
    print(count_list)


def main(args):
    result_list = []
    for file_dir in args.predict_dir:
        if args.dev:
            file_dir = os.path.join(file_dir, 'dev_result')
        else:
            file_dir = os.path.join(file_dir, 'predict_result')

        file_list = os.listdir(file_dir)
        method_list = []
        for file in file_list:
            if '_right_result.json' in file:
                method = file.replace('_right_result.json', '')
                method_list.append(method)
            elif '_predict.json' in file:
                method = file.replace('_predict.json', '')
                method_list.append(method)
        
        if len(method_list) == 0:
            result_list.append((load_result(args, file_dir, method=None), f"{file_dir} [origin]"))
        else:
            for method in method_list:
                result_list.append((load_result(args, file_dir, method=method), f"{file_dir} [{method}]"))

    if args.task_name == 'merge':
        right_dict = {}
        for index, result in enumerate(result_list):
            info = result[0]['info']
            print(f"{index} - {result[1]}\nright [{info['right']}]; acc [{info['acc']:.7}]")
            right_dict.update(result[0]['right_dict'])

        total_right = len(right_dict)
        acc = total_right / 1221 * 100
        print(f"After merge, right [{total_right}]; acc [{acc:.4f}]")

    elif args.task_name in ['vote', 'vote_logit']:
        csqa_dict = count_vote(args, result_list, args.choose_top, args.task_name)

        if not args.dev:
            f = open('predict.json', 'w', encoding='utf-8')
            csqa_list = []
            for k, v in csqa_dict.items():
                v['id'] = k
                v['AnswerKey_pred'] = v['predict']
                del v['predict']
                csqa_list.append(v)
            json.dump(csqa_list, f, indent=2)
            f.close()
            

    elif args.task_name == "compare":
        csqa_dict = count_vote(args, result_list, False, args.task_name)
        compare(args, csqa_dict, result_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=['vote_logit', 'merge', 'vote', 'compare'])
    parser.add_argument('--predict_dir', nargs='+')
    parser.add_argument('--choose_top', action='store_true')
    parser.add_argument('--dev', action='store_true')

    # --dev
    # --choose_top
    # --dev
    args_str = r"""
    --task_name vote
    --predict_dir 
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\WKDT\1829-May04_seed5004_wkdtv4.0_80.59
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\WKDT\1138-Apr23_seed42_wkdtv3.0
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\origin\model_01_80.01
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\origin\model_03_80.10
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\MSKE_OMCS\train02_equal_dev5_group
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\MSKE_OMCS\train02_train02_80.26
    """
    # D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\BurgerAlpha2\layer11_80.18
    # D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\origin\model_02_77.96
    # D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\WKDT\1312-May11_seed5004_wkdtv5.0_75.53

    # args_str = """
    # --task_name vote
    # --predict_dir 
    # D:\CODE\Commonsense\CSQA_DATA\model_save\1528-Apr22_seed42\dev_result
    # /data/zhifli/model_save/albert-xxlarge-v2/Origin_Albert_Baseline/1712-Apr09_seed42/
    # /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1829-May04_seed5004_wkdtv4.0/
    # /data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/1138-Apr23_seed42_wkdtv3.0/
    # /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1311-Apr26_seed425_cs1_omcsv3.0/
    # /data/zhifli/model_save/albert-xxlarge-v2/OMCS_Albert_Baseline/1312-Apr26_seed42_cs2_omcsv3.0/
    # """

    args = parser.parse_args(args_str.split())
    print(args)
    # args = parser.parse_args()

    main(args)
    
