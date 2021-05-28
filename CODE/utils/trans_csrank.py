#! -*- encoding:utf-8 -*-
"""
@File    :   trans_csranktoomcs.py
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

def load_omcs(dataset_dir, omcs_version, target):
    dir_dict = {'1.0':'omcs_v1.0', '3.0':'omcs_v3.0_15', '3.1':'omcs_v3.1_10'}

    omcs_file = os.path.join(dataset_dir, 'omcs', dir_dict[omcs_version] ,f"{target}_rand_split_omcs.json")

    with open(omcs_file, 'r', encoding='utf-8') as f:
        omcs_cropus = json.load(f)

    return omcs_cropus

def load_rank(data_dir, target):

    file_dir = os.path.join(data_dir, f"{target}_csrank.json")
    f = open(file_dir, 'r', encoding='utf-8')
    csrank_data = json.load(f)
    f.close()
    return csrank_data

def rank_omcs(omcs_data, csrank_data):
    for omcs, rank_omcs in zip(omcs_data, csrank_data):
        cs_list = [cs[1] for cs in rank_omcs['cs_list']]
        logit_list = [cs[0] for cs in rank_omcs['cs_list']]
        omcs['cs_list'] = cs_list
        omcs['logit_list'] = logit_list
    return omcs_data

def dump_omcs(dataset_dir, version, omcs_data, target):
    data_dir = os.path.join(dataset_dir, 'omcs', f'omcs_v{version}_rank', f"{target}_rand_split_omcs.json")
    mkdir_if_notexist(data_dir)
    f = open(data_dir, 'w', encoding='utf-8')
    json.dump(omcs_data, f, indent=4)
    f.close()

def inject_wkdt(csqa_data, wkdt_rank_data):
    wkdt_data = []
    for case_index, case in enumerate(csqa_data):
        wkdt_current = wkdt_rank_data[case_index * 5:case_index * 5+5]
        # if case['id'] != wkdt_current[0]['id']:
        question = case['question']
        stem = question['stem']
        question_concept = question['question_concept']
        choices = question['choices']
        for choice_index, choice in enumerate(choices):
            case_dict = {
                'id': case['id'],
                'question': stem,
                'question_concept': question_concept,
                'choice': choice['text'],
                'answerKey': case['answerKey'],
                'cs_list':[],
                'logit_list':[]
            }
            wkdt = wkdt_current[choice_index]['desc_list']
            for desc_group in wkdt[:4]:
                logit, desc = desc_group
                desc = [desc['Qdesc'], desc['Cdesc']]
                case_dict['cs_list'].append(desc)
                case_dict['logit_list'].append(logit)
        
            wkdt_data.append(case_dict)

    return wkdt_data

def dump_wkdt(dataset_dir, version, csqa_wkdt, target):
    data_dir = os.path.join(dataset_dir, 'wkdt', f'wiktionary_v{version}_rank', f"{target}_concept.json")
    mkdir_if_notexist(data_dir)
    f = open(data_dir, 'w', encoding='utf-8')
    json.dump(csqa_wkdt, f, indent=4)
    f.close()                

    
if __name__ == "__main__":
    # target = 'omcs'
    data_dir = "/home/zhifli/DATA/omcs/omcs_v3.0_rank/cs_rank/"
    dataset_dir = "/home/zhifli/DATA/"

    target = 'wkdt'
    data_dir = "/data/zhifli/model_save/albert-xxlarge-v2/WKDT_Albert_Baseline/wkdtv4.0_colab/"
    dataset_dir = "/home/zhifli/DATA/"

    if target == 'omcs':
        for target in ["dev", "train", "test"]:
            omcs_data = load_omcs(dataset_dir, "3.0", target)
            csrank_data = load_rank(data_dir, target)
            omcs_data = rank_omcs(omcs_data, csrank_data)
            dump_omcs(dataset_dir, "3.0", omcs_data, target)
    elif target == 'wkdt':
        for target in ["dev", "train"]:
            csqa_data = load_csqa(dataset_dir, target)
            wkdt_rank_data = load_rank(data_dir, target)
            csqa_wkdt = inject_wkdt(csqa_data, wkdt_rank_data)
            dump_wkdt(dataset_dir, "5", csqa_wkdt, target)
            
