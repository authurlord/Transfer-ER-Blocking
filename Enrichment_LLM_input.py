import pandas as pd
import numpy as np
import json
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import argparse
parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")

parser.add_argument("--target_data", type=str)

args = parser.parse_args()
target_data = args.target_data
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

attribute_set_dict = {
    'AB':['category','sku','brand','modelno','key_features'],
    'WA':['subcategory','key_features','sku','color'],
    'AG':['category','subcategory','platform','edition','type','modelno'],
    'DA':['keywords','research area'],
    'DS':['keywords','research area'],
}

tableA_T_path = annotation_dict[target_data]['tableA_path']
tableB_T_path = annotation_dict[target_data]['tableB_path']
train_T_path = annotation_dict[target_data]['train_path']
valid_T_path = annotation_dict[target_data]['valid_path']
test_T_path = annotation_dict[target_data]['test_path'] 

tableA_T = pd.read_csv(tableA_T_path).fillna('')
tableB_T = pd.read_csv(tableB_T_path).fillna('')
tableA_dict_T = tableA_T.set_index('id').to_dict(orient='records')
tableB_dict_T = tableB_T.set_index('id').to_dict(orient='records')
train_T = pd.read_csv(train_T_path)
valid_T = pd.read_csv(valid_T_path)
test_T = pd.read_csv(test_T_path)
all_value_T = pd.concat([train_T,valid_T,test_T]).reset_index(drop=True)


format_dict = {}
format_dict['Entity 1'] = {}
format_dict['Entity 2'] = {}
attr_list = list(tableA_dict_T[0].keys())
attr_list.extend(attribute_set_dict[target_data])

for attr in attr_list:
    format_dict['Entity 1'][attr] = ''
    format_dict['Entity 2'][attr] = ''
    
def Transfer_Text_To_Dict(row): ## For Walmart-Amazon Transfer
    format_head = 'Enrich Entity 1 and Entity 2 with attributes: %s. Return in json format.' % '/'.join(attribute_set_dict[target_data])
    ltable_id = int(row['ltable_id'])
    rtable_id = int(row['rtable_id'])
    Entity_1_dict = tableA_dict_T[ltable_id] ## content of Entity 1
    Entity_1 = 'Entity 1:%s' % json.dumps(Entity_1_dict)
    Entity_2_dict = tableB_dict_T[rtable_id] ## content of Entity 2
    Entity_2 = 'Entity 2:%s' % json.dumps(Entity_2_dict)
    format_example = 'Output Format Example:\n\n%s\n\n' % json.dumps(format_dict) 
    return format_head + '\n\n' + format_example + Entity_1 + '\n\n' + Entity_2
all_value_T_unlabel = all_value_T.iloc[:,:2]
all_value_T_unlabel['text_mistral'] = all_value_T_unlabel.progress_apply(Transfer_Text_To_Dict,axis=1)

all_value_T_unlabel.to_csv('enrich_data/enrich_query/%s_input.csv' % target_data)