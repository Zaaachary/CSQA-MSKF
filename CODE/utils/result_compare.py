import argparse
import os
import json
from collections import Counter
from common import mkdir_if_notexist

def load_csqa(dataset_dir="/home/zhifli/DATA"):
    raw_csqa = []
    f = open(os.path.join(dataset_dir, 'csqa', "dev_rand_split.jsonl"), 'r', encoding='utf-8')
    for line in f:
        raw_csqa.append(json.loads(line.strip()))
    f.close()
    return raw_csqa

def result_dump(args, target, file_name, folder=''):
    
    mkdir_if_notexist(os.path.join(args.result_dir, folder, file_name))
    with open(os.path.join(args.result_dir, folder, file_name), 'w', encoding='utf-8') as f:
        json.dump(target, f, ensure_ascii=False, indent=4)

def load_result(model_dir, method=None):

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


    return result

def count_result(args, result_list, task_name):
    csqa_dict = {}

    origin = result_list.pop(0)
    origin = origin[0]
    origin['right_dict'].update(origin['wrong_dict'])

    for key, value in origin['right_dict'].items():
        csqa_dict[key] = {
            'answerKey': value['answerKey'],
            'predictList': [value['AnswerKey_pred'], ],
            'logitsList': [[choice['logit'] for choice in value['choices']], ]
        }

    for result in result_list:
        result = result[0]
        result['right_dict'].update(result['wrong_dict'])

        for key, value in result['right_dict'].items():
            csqa_dict[key]['predictList'].append(value['AnswerKey_pred'])
            csqa_dict[key]['logitsList'].append([choice['logit'] for choice in value['choices']])
    return csqa_dict


def compare_v1(args, result_list):
    # 找到 model1做错  model2 做对的项
    result = {}
    model1, model2 = result_list
    wrong1, right2 = model1[0]["wrong_dict"], model2[0]["right_dict"]
    for key, value in wrong1.items():
        if key in right2:
            result[key] = [value, right2[key]]
    return result

def get_question_type(args):
    raw_csqa = load_csqa(args.dataset_dir)
    question = {
        'what': [],
        'where': [],
        'why': [],
        'how': [],
        'others': []
    }
    for case in raw_csqa:
        stem = case['question']['stem'].lower()
        for target in ['why', 'how', 'where', 'what']:
            if target in stem:
                question[target].append(case['id'])
                break
        else:
            question['others'].append(case['id'])
    
    for key, value in question.items():
        print(key, f": {len(value)}")

    return question

def count_acc(qeustion_type, result_list):
    # import pdb; pdb.set_trace()
    total_acc = []
    for rindex, result in enumerate(result_list):
        question_count = {key: 0 for key in qeustion_type.keys()}
        data, model_dir = result
        right = data['right_dict']
        for key in right.keys():
            for k, v in qeustion_type.items():
                if key in v:
                    question_count[k] += 1
                    break
        total_acc.append(question_count)
    return total_acc

def show_acc(total_acc, question_type):
    for qtype, value in question_type.items():
        qtype_num = len(value)
        print("qtype:", qtype)
        for index, acc in enumerate(total_acc):
            print(f"{index}: {acc[qtype]}/{len(value)} => {acc[qtype]/len(value):0.4f}", )
    

def main(args):
    # load result
    result_list = []
    for file_dir in args.predict_dir:
        file_dir = os.path.join(file_dir, 'dev_result')

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
            result_list.append((load_result(file_dir, method=None), f"{file_dir} [origin]"))
        else:
            for method in method_list:
                result_list.append((load_result(file_dir, method=method), f"{file_dir} [{method}]"))

    if args.task_name == "compare":
        result = compare_v1(args, result_list)
        result_dump(args, result, f'result_{len(result)}.json')
    elif args.task_name == 'question_type':
        question_type = get_question_type(args)
        total_acc = count_acc(question_type, result_list)
        show_acc(total_acc, question_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, choices=['compare', 'question_type'])
    parser.add_argument('--predict_dir', nargs='+')
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--dataset_dir', type=str)

    args_str = r"""
    --task_name question_type
    --dataset_dir D:\CODE\Commonsense\CSQA\DATA
    --predict_dir
    D:\CODE\Commonsense\CSQA_DATA\model_save\xxlarge\origin\model_03_80.10
    D:\CODE\Commonsense\CSQA_DATA\model_save\OMCS\1946-May21_seed42_cs3_omcsv3.0_rank_80.99%
    D:\CODE\Commonsense\CSQA_DATA\model_save\WKDT\1829-May04_seed5004_wkdtv4.0_80.59
    D:\CODE\Commonsense\CSQA_DATA\predict\MSKF_84.36

    --result_dir
    D:\CODE\Commonsense\CSQA_DATA\model_save\compare\
    """

    args = parser.parse_args(args_str.split())
    # args = parser.parse_args()
    print(args, '\n')

    main(args)
    
