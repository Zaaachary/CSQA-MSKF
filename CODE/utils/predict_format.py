#! -*- encoding:utf-8 -*-
"""
@File    :   predict_format.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   

[
    {
        "id"
        "AnswerKey_pred"
    },
]

=> 
"""
import json
import pandas

raw_prediction_dir = r"D:\CODE\Commonsense\CSQA_dev\DATA\result\prediction"

origin_dir = raw_prediction_dir + r"\omcs_3.0_predict.json"
# origin_dir = r"D:\CODE\Commonsense\CSQA_dev\DATA\result\albert-base-v2\WKDT_Albert_Baseline\2238-Apr19_seed42_wkdtv3.0\predict.json"
output_dir = raw_prediction_dir + r"\questions.json"

output_dir2 = raw_prediction_dir + r"\predictions.csv"
# output_dir = r"D:\CODE\Commonsense\CSQA_dev\DATA\result\albert-base-v2\WKDT_Albert_Baseline\2238-Apr19_seed42_wkdtv3.0\predict_format.json"

def load_result():
    f = open(origin_dir, 'r', encoding='utf-8')
    all_result = json.load(f)
    f.close()

    all_case = []
    for case in all_result:
        q_id = case['id']
        answerKey = case['AnswerKey_pred']
        
        new_case = {'id':q_id, "answerKey":answerKey}
        all_case.append(new_case)

    return all_case

def create_jsonl(all_case):
    f = open(output_dir, 'w', encoding='utf-8')
    for case in all_case:
        s = json.dumps(case)
        f.write(s+'\n')
    f.close()

# def create_csv(all_case):
#     f = open(output_dir2, 'w', encoding='utf-8')
#     for case in all_case:
#         s = f"{case['id'],case['answer_Key']}"
#         f.write(s+'\n')
#     f.close()


if __name__ == "__main__":
    all_case = load_result()
    create_jsonl(all_case)
    create_csv(all_case)

