import numpy as np
import pandas as pd 
from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')

import random
from tqdm.notebook import tqdm
import copy

import argparse
from types import SimpleNamespace
import os 
from json_repair import repair_json
import json

parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")

parser.add_argument("--data_name", type=str)
parser.add_argument("--model_name", type=str)

args = parser.parse_args()

def extract_and_wrap(text):
    # 查找第一个 '{' 和最后一个 '}' 的索引
    start_index = text.find('{')
    end_index = text.rfind('}')

    # 检查是否有匹配的 '{' 和 '}'
    if start_index == -1 or end_index == -1 or start_index > end_index:
        return "No matching braces found."

    # 提取第一个 '{' 和最后一个 '}' 之间的内容
    extracted_content = text[start_index + 1:end_index]

    # 在头部和尾部分别补上 '{' 和 '}'
    wrapped_content = "{" + extracted_content + "}"

    return wrapped_content
def extract_and_wrap_multi_dict(text):
    # 查找第一个 '{' 和最后一个 '}' 的索引
    start_index = text.rfind('\n\n{"Entity 1')
    end_index = text.rfind('}')

    # 检查是否有匹配的 '{' 和 '}'
    if start_index == -1 or end_index == -1 or start_index > end_index:
        return "No matching braces found."

    # 提取第一个 '{' 和最后一个 '}' 之间的内容
    extracted_content = text[start_index + 1:end_index]

    # 在头部和尾部分别补上 '{' 和 '}'
    wrapped_content = extracted_content + "}"

    return wrapped_content

data_name = args.data_name

model_name = args.model_name

enrich_query_path = 'enrich_data/enrich_query/%s_input.csv' % data_name


enrich_query = pd.read_csv(enrich_query_path,index_col=0)
enrich_query['ltable_enrich'] = ''
enrich_query['rtable_enrich'] = ''

LLM_output = np.load('enrich_data/enrich_query/%s_output.npy' % data_name)

annotation_dict = {
    'AG':{
        'tableA_path':'process_data/amazon-google/amazon-google/tableA.csv',
        'tableB_path':'process_data/amazon-google/amazon-google/tableB.csv',
        'train_path' : 'process_data/amazon-google/amazon-google/train.csv',
        'valid_path' : 'process_data/amazon-google/amazon-google/valid.csv',
        'test_path' : 'process_data/amazon-google/amazon-google/test.csv',
    },
    'WA':{
        'tableA_path':'process_data/walmart-amazon/exp_data/tableA.csv',
        'tableB_path':'process_data/walmart-amazon/exp_data/tableB.csv',
        'train_path' : 'process_data/walmart-amazon/exp_data/train.csv',
        'valid_path' : 'process_data/walmart-amazon/exp_data/valid.csv',
        'test_path' : 'process_data/walmart-amazon/exp_data/test.csv',
    },
    'AB':{
        'tableA_path':'process_data/ant-buy/exp_data/tableA.csv',
        'tableB_path':'process_data/ant-buy/exp_data/tableB.csv',
        'train_path' : 'process_data/ant-buy/exp_data/train.csv',
        'valid_path' : 'process_data/ant-buy/exp_data/valid.csv',
        'test_path' : 'process_data/ant-buy/exp_data/test.csv',
    },
    'RI':{
        'table_path':'process_data/RI/all_value.csv',
    },
    'IA':{
        'tableA_path':'process_data/iTunes/iTunes/tableA.csv',
        'tableB_path':'process_data/iTunes/iTunes/tableB.csv',
        'train_path' : 'process_data/iTunes/iTunes/train.csv',
        'valid_path' : 'process_data/iTunes/iTunes/valid.csv',
        'test_path' : 'process_data/iTunes/iTunes/test.csv',
    },
    'DA':{
        'tableA_path':'process_data/dblp-acm/exp_data/tableA.csv',
        'tableB_path':'process_data/dblp-acm/exp_data/tableB.csv',
        'train_path' : 'process_data/dblp-acm/exp_data/train.csv',
        'valid_path' : 'process_data/dblp-acm/exp_data/valid.csv',
        'test_path' : 'process_data/dblp-acm/exp_data/test.csv',
    },
    'DS':{
        'tableA_path':'process_data/dblp-google/exp_data/tableA.csv',
        'tableB_path':'process_data/dblp-google/exp_data/tableB.csv',
        'train_path' : 'process_data/dblp-google/exp_data/train.csv',
        'valid_path' : 'process_data/dblp-google/exp_data/valid.csv',
        'test_path' : 'process_data/dblp-google/exp_data/test.csv',
    },
}

count = 0
for i in tqdm(range(len(LLM_output))):
    text = LLM_output[i]
    if(text.__contains__('json')): ## 有```json```,只看```json包裹内的内容
        a = repair_json(text.split('```json')[-1].split('```')[0])
    else:
        a = repair_json(extract_and_wrap(text)) ##匹配第一个{与最后一个}
    try:
        dict_ouput = json.loads(a)
        enrich_query.iloc[i,-2] = str(dict_ouput['Entity 1'])
        enrich_query.iloc[i,-1] = str(dict_ouput['Entity 2'])
        dict_ouput['Entity 1']
    except:
        try:
            a = repair_json(extract_and_wrap_multi_dict(text)) ## 匹配最后一个\n\n{"Entity 1 与最后一个'
            dict_ouput = json.loads(a)
            enrich_query.iloc[i,-2] = str(dict_ouput['Entity 1'])
            enrich_query.iloc[i,-1] = str(dict_ouput['Entity 2'])
        except:
            count += 1
print(count / len(LLM_output)) ## 解析失败率


### Merge in Dicts

tableA_T_path = annotation_dict[data_name]['tableA_path']
tableB_T_path = annotation_dict[data_name]['tableB_path']
tableA_T = pd.read_csv(tableA_T_path).fillna('')
tableB_T = pd.read_csv(tableB_T_path).fillna('')
tableA_dict_T = tableA_T.set_index('id').to_dict(orient='records')
tableB_dict_T = tableB_T.set_index('id').to_dict(orient='records')
dict_ltable = {}
dict_rtable = {}
for key in range(len(tableA_dict_T)):
    dict_ltable[key] = [tableA_dict_T[key]] ## entity属性扩展成列表，容纳更多的enrich选项
for key in range(len(tableB_dict_T)):
    dict_rtable[key] = [tableB_dict_T[key]] ## entity属性扩展成列表，容纳更多的enrich选项

for index,row in enrich_query.iterrows():
   
    
    ltable_id = int(row[0])
    rtable_id = int(row[1])
    
    if(row[-2]!=''): ## 解析成功
        ltable_enrich = eval(row[-2]) ## 解析回字典，因为存储是Str
        temp_l = dict_ltable[ltable_id]
        if ltable_enrich not in temp_l: ## 避免重复
            temp_l.append(ltable_enrich)
        dict_ltable[ltable_id] = temp_l
    if(row[-1]!=''): ## 解析成功
        rtable_enrich = eval(row[-1]) ## 解析回字典，因为存储是Str
        temp_r = dict_rtable[rtable_id]
        if rtable_enrich not in temp_r: ## 避免重复
            temp_r.append(rtable_enrich)
        dict_rtable[rtable_id] = temp_r
        
np.save('enrich_data/enrich_dict/dict_ltable_%s_%s.npy' % (data_name,model_name),dict_ltable)
np.save('enrich_data/enrich_dict/dict_rtable_%s_%s.npy' % (data_name,model_name),dict_rtable) 

