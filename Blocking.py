import numpy as np
import pandas as pd 
from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')
from pandarallel import pandarallel
import random
from tqdm.notebook import tqdm
import copy
pandarallel.initialize(progress_bar=True)
import subprocess
import argparse
from types import SimpleNamespace
import os 
from json_repair import repair_json
import json
import torch
from FlagEmbedding import FlagModel

parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")
parser.add_argument("--source_data", type=str)
parser.add_argument("--target_data", type=str)
parser.add_argument("--backbone_model_name", type=str)
parser.add_argument("--pseudo_label_thres", type=float,default=0.95)
parser.add_argument("--sbert_model_path", type=str, default='bge-large-en-1.5')
# parser.add_argument("--similarity_path", type=str, default='', help="similarity matrix save path")
# parser.add_argument("-wdc", "--wdc", action="store_true", help="Enable verbose mode")
args = parser.parse_args()


import random

def select_dict(dicts): ## 输入一个list，选择其中enrich后的一个dict；如果只有一个元素，则输出这个元素自己
    # 检查输入是否只有一个元素
    if len(dicts) == 1:
        return dicts[0]
    
    # 筛选出键数量大于3的字典
    filtered_dicts = [d for d in dicts if len(d.keys()) > 3]
    
    # 如果有符合条件的字典，随机返回一个
    if filtered_dicts:
        return random.choice(filtered_dicts)
    else:
        # 如果没有符合条件的字典，可以返回None或其他值
        return None
def matrix_multiply_gpu(A_np, B_np): ## 使用torch的矩阵乘法运算
    # 确保 PyTorch 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将 NumPy 数组转换为 PyTorch 张量
    A_torch = torch.tensor(A_np).to(device)
    B_torch = torch.tensor(B_np).to(device)

    # 执行矩阵乘法
    result_torch = torch.matmul(A_torch, B_torch.t())

    # 将结果转换回 NumPy 数组
    result_np = result_torch.cpu().numpy()

    return result_np
def merge_dict_lists(input_dict, keys_list): ## RAG对比学习使用函数，输入一个dict和index list，将其中所有dict合并为一个list，并转为str
    # 初始化一个空列表，用于存储合并后的结果
    merged_list = []

    # 遍历 keys_list 中的每个键
    for key in keys_list:
        # 检查键是否在 input_dict 中
        if key in input_dict:
            # 遍历 input_dict[key] 中的每个字典
            for item in input_dict[key]:
                # 确保每个 item 是一个字典，并且将其转换为字符串格式
                if isinstance(item, dict):
                    merged_list.append(str(item))
    
    return merged_list
import hashlib

## 计算文件的sha1值
def encrypt(fpath: str, algorithm: str) -> str:
    with open(fpath, 'rb') as f:
        return hashlib.new(algorithm, f.read()).hexdigest()
def compare_dataframes(df1, df2):
    # 取出前两列
    df1_subset = df1.iloc[:, :2]
    df2_subset = df2.iloc[:, :2]
    
    # 重置索引以确保比较时索引不影响结果
    df1_subset_reset = df1_subset.reset_index()
    df2_subset_reset = df2_subset.reset_index()
    
    # 合并两个DataFrame，使用前两列进行比较
    merged_df = pd.merge(df1_subset_reset, df2_subset_reset, on=df1_subset.columns.tolist(), how='inner')
    
    # 提取重合行的索引
    common_indices_df1 = merged_df['index_x']
    common_indices_df2 = merged_df['index_y']

    return common_indices_df1, common_indices_df2
def replace_rows_if_match(df1, df2): ## df1是原始的CL结果，df2是LLM额外标注的结果，将额外标注的结果取代原有的CL结果
    # 创建一个副本以避免修改原始 DataFrame
    df1_copy = df1.copy()

    # 遍历 df1 的每一行
    for i, row in df1.iterrows():
        # 获取 df1 当前行的前两列值
        row_values = row.iloc[:2].values

        # 查找 df2 中是否有匹配的行
        matched_rows = df2[(df2.iloc[:, 0] == row_values[0]) & (df2.iloc[:, 1] == row_values[1])]

        # 如果找到匹配的行，则替换 df1 中的当前行
        if not matched_rows.empty:
            df1_copy.iloc[i] = matched_rows.iloc[0]

    return df1_copy
import hashlib
import json
import os

def calculate_sha1(file_path):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()

def generate_dict(file_list): ## 生成输入dataset_info的注册信息
    result = {}
    for file_path in file_list:
        # 计算文件的 SHA-1 值
        file_sha1 = calculate_sha1(file_path)
        
        # 生成键名，假设文件名格式为 "walmart_amazon_train_output_r1.json"
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        if len(parts) > 1:
            key_name = f"Transfer-ER-{parts[0]}-{parts[1]}-{parts[-1].split('.')[0]}"
        else:
            key_name = f"Transfer-ER-{file_name.split('.')[0]}"
        
        # 构建字典值
        result[key_name] = {
            "file_name": '/data/home/wangys/LLM_ER/' + file_path,
            "file_sha1": file_sha1,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    
    return result

def list_to_json(file_list): ## 生成输入dataset_info的注册信息
    result_dict = generate_dict(file_list)
    return json.dumps(result_dict, indent=2)

import random

def random_choice_exclude(a, b, k):
    """
    从列表 a 中随机选择 k 个元素，避开列表 b 中的所有元素。
    
    参数:
    a (list): 候选列表
    b (list): 需要避开的元素列表
    k (int): 需要选择的元素个数
    
    返回:
    list: 从列表 a 中选择的 k 个元素列表
    """
    # 使用集合进行差集操作以提高效率
    available_elements = list(set(a) - set(b))
    
    # 如果可用元素不足 k 个，抛出异常
    if len(available_elements) < k:
        raise ValueError("可用元素不足以选择 k 个元素")
    
    # 从可用元素中随机选择 k 个元素
    return random.sample(available_elements, k)
def filter_list_with_set(original_list, filter_set):
    """
    从原始列表中筛选出所有在筛选集合中的元素，保持原有顺序。

    参数:
    original_list (list): 原始列表
    filter_set (set): 筛选集合

    返回:
    list: 筛选后的列表
    """
    return list(filter(lambda x: x in filter_set, original_list))
from collections import Counter

def most_common_values(list_of_dicts):
    """
    从一个包含多个字典的列表中，提取每个键出现次数最多的值。
    如果列表中只有一个字典，则直接输出该字典。

    参数:
    list_of_dicts (list): 包含多个字典的列表，每个字典的值均看作字符串

    返回:
    dict: 每个键及其出现次数最多的值组成的字典
    """
    # 检查列表中是否只有一个字典
    if(isinstance(list_of_dicts,dict)):
        return list_of_dicts
    if len(list_of_dicts) == 1:
        return list_of_dicts[0]
    
    # 初始化一个空的字典来存储每个键及其对应的值的计数
    value_counts = {}
    
    # 遍历列表中的每个字典
    for d in list_of_dicts:
        if(isinstance(d,dict)):
            for key, value in d.items():
                # 确保值是字符串
                value = str(value)
                if key not in value_counts:
                    value_counts[key] = Counter()
                value_counts[key][value] += 1
    
    # 从计数中提取每个键的最常见值
    most_common_dict = {key: counter.most_common(1)[0][0] for key, counter in value_counts.items()}
    
    return most_common_dict
import numpy as np

def find_elements_above_threshold(matrix, threshold):
    """
    查找矩阵中所有大于给定阈值的元素，并返回它们的行列索引。

    参数:
    matrix (list of list of numbers): 输入的矩阵
    threshold (number): 阈值

    返回:
    list of tuple: 每个元素是一个元组，包含行索引和列索引
    """
    # 将输入的矩阵转换为numpy数组，便于操作
    np_matrix = np.array(matrix)
    
    # 找到大于阈值的元素的索引
    indices = np.argwhere(np_matrix > threshold)
    
    # 将numpy数组中的索引转换为Python列表中的元组
    result = [tuple(index) for index in indices]
    
    return result


model_name_suffix_dict = {
    'mistral-7B':'mistral',
    'llama3-8B' :'llama3',
    'qwen2-1.5B' : 'qwen2-1_5B'
}

align_thres_dict = {
    'AB-AG':0.8,
    'WA-AG':0.92,
    'WA-AB':0.93,
    'AB-WA':0.93,
    'RI-WA':0.74,
    'RI-AB':0.7,
    'IA-DA':0.77,
    'IA-DS':0.77
}
pseudo_label_thres = args.pseudo_label_thres
model_name = args.backbone_model_name
source_data = args.source_data
data_name_S = source_data
target_data = args.target_data
data_name_T = target_data
model_name_suffix = model_name_suffix_dict[args.backbone_model_name]

align_pair_name = '%s-%s-%s' % (source_data,target_data,model_name_suffix)

align_pair = '%s-%s' % (source_data,target_data)
align_json_file = 'temp_data/align/%s-align.json' % align_pair_name
CL_file = 'temp_data/CL/%s-CL.json' % align_pair_name
align_thres = align_thres_dict[align_pair]

bge_model_path = args.sbert_model_path

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

dict_path = {
    'AG':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_amazon_google_%s.npy' % model_name_suffix,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_amazon_google_%s.npy' % model_name_suffix,
    },
    'AB':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_ant_buy_%s.npy' % model_name_suffix,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_ant_buy_%s.npy' % model_name_suffix,
    },
    'WA':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_walmart_amazon_%s.npy' % model_name_suffix,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_walmart_amazon_%s.npy' % model_name_suffix,
    },
    'DA':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_dblp_acm_%s.npy' % model_name_suffix,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_dblp_acm_%s.npy' % model_name_suffix,
    },
    'DS':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_dblp_google_%s.npy' % model_name_suffix,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_dblp_google_%s.npy' % model_name_suffix,
    },
    'RI':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_RI.npy' ,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_RI.npy' ,
    },
    'IA':{
        'dict_ltable':'enrich_data/enrich_dict/dict_ltable_IA.npy' ,
        'dict_rtable':'enrich_data/enrich_dict/dict_rtable_IA.npy' ,
    },
}

### Loading source and target table



if source_data in ['WA','AG','AB','IA']: ## Standard Format
    tableA_S_path = annotation_dict[source_data]['tableA_path']
    tableB_S_path = annotation_dict[source_data]['tableB_path']
    train_S_path = annotation_dict[source_data]['train_path']
    valid_S_path = annotation_dict[source_data]['valid_path']
    test_S_path = annotation_dict[source_data]['test_path'] 

    dict_ltable_path_S = dict_path[source_data]['dict_ltable']
    dict_rtable_path_S = dict_path[source_data]['dict_rtable']
    
    tableA_S = pd.read_csv(tableA_S_path).fillna('')
    tableB_S = pd.read_csv(tableB_S_path).fillna('')
    tableA_dict_S = tableA_S.set_index('id').to_dict(orient='records')
    tableB_dict_S = tableB_S.set_index('id').to_dict(orient='records')
    train_S = pd.read_csv(train_S_path)
    valid_S = pd.read_csv(valid_S_path)
    test_S = pd.read_csv(test_S_path)
    try: ## Not Include IA
        dict_ltable_S = np.load(dict_ltable_path_S,allow_pickle=True).item()
        dict_rtable_S = np.load(dict_rtable_path_S,allow_pickle=True).item()
    except: ## IA
        dict_ltable_S = np.load(dict_ltable_path_S,allow_pickle=True)
        dict_rtable_S = np.load(dict_rtable_path_S,allow_pickle=True)
    all_value_S = pd.concat([train_S,valid_S,test_S]).reset_index(drop=True)
elif source_data in ['RI']:
    dict_ltable_path_S = dict_path[source_data]['dict_ltable']
    dict_rtable_path_S = dict_path[source_data]['dict_rtable']
    table_S_path = annotation_dict[source_data]['table_path'] 

    dict_ltable_S = np.load(dict_ltable_path_S,allow_pickle=True).item()
    dict_rtable_S = np.load(dict_rtable_path_S,allow_pickle=True).item()
    all_value_S = pd.read_csv(table_S_path,index_col=0)
    
tableA_T_path = annotation_dict[target_data]['tableA_path']
tableB_T_path = annotation_dict[target_data]['tableB_path']
train_T_path = annotation_dict[target_data]['train_path']
valid_T_path = annotation_dict[target_data]['valid_path']
test_T_path = annotation_dict[target_data]['test_path'] 
dict_ltable_path_T = dict_path[target_data]['dict_ltable']
dict_rtable_path_T = dict_path[target_data]['dict_rtable']

tableA_T = pd.read_csv(tableA_T_path).fillna('')
tableB_T = pd.read_csv(tableB_T_path).fillna('')
tableA_dict_T = tableA_T.set_index('id').to_dict(orient='records')
tableB_dict_T = tableB_T.set_index('id').to_dict(orient='records')
train_T = pd.read_csv(train_T_path)
valid_T = pd.read_csv(valid_T_path)
test_T = pd.read_csv(test_T_path)
dict_ltable_T = np.load(dict_ltable_path_T,allow_pickle=True).item()
dict_rtable_T = np.load(dict_rtable_path_T,allow_pickle=True).item()
all_value_T = pd.concat([train_T,valid_T,test_T]).reset_index(drop=True)
all_value_pos_T = all_value_T[all_value_T['label']==1].reset_index(drop=True) ## all_value_pos是ground truth，只做evaluation用


### Self-Supervised Alignment

model = FlagModel(bge_model_path,use_fp16=True)
embedding_tableA_S = model.encode([str(most_common_values(dict_ltable_S[i])) for i in range(len(dict_ltable_S))]) 
embedding_tableB_S = model.encode([str(most_common_values(dict_rtable_S[i])) for i in range(len(dict_rtable_S))]) 
embedding_tableA_T = model.encode([str(most_common_values(dict_ltable_T[i])) for i in range(len(dict_ltable_T))])
embedding_tableB_T = model.encode([str(most_common_values(dict_rtable_T[i])) for i in range(len(dict_rtable_T))])
similarity_A_A_ST = matrix_multiply_gpu(embedding_tableA_S , embedding_tableA_T)
similarity_A_B_ST = matrix_multiply_gpu(embedding_tableA_S , embedding_tableB_T)
similarity_B_A_ST = matrix_multiply_gpu(embedding_tableB_S , embedding_tableA_T)
similarity_B_B_ST = matrix_multiply_gpu(embedding_tableB_S , embedding_tableB_T)

## Create Self-Supervised Alignment

with open(align_json_file , 'w', encoding='utf-8') as file:
    for (source,target) in tqdm(find_elements_above_threshold(similarity_B_A_ST,align_thres)):
        l_dict = most_common_values(dict_rtable_S[source]) ## 注意similarity_B_A_ST
        r_dict = most_common_values(dict_ltable_T[target])
        positive_list = [str(r_dict)]
        # print(len(str(r_dict)))
        # positive_list.extend(merge_dict_lists(dict_ltable_T,[target]))
        data = {
            "query": str(l_dict),
            "pos" : positive_list,
            "neg" : []
            
        }
    # 将字典转换为JSON字符串，并写入文件
        file.write(json.dumps(data, ensure_ascii=False))
        file.write('\n')  # 每个JSON对象后添加换行符
    for (source,target) in tqdm(find_elements_above_threshold(similarity_B_B_ST,align_thres)):
        l_dict = most_common_values(dict_rtable_S[source]) ## 注意similarity_B_B_ST
        r_dict = most_common_values(dict_rtable_T[target])
        positive_list = [str(r_dict)]
        # print(len(str(r_dict)))
        # positive_list.extend(merge_dict_lists(dict_rtable_T,[target]))
        data = {
            "query": str(l_dict),
            "pos" : positive_list,
            "neg" : []
            
        }
    # 将字典转换为JSON字符串，并写入文件
        file.write(json.dumps(data, ensure_ascii=False))
        file.write('\n')  # 每个JSON对象后添加换行符
print(align_json_file)
## Hard Negative Sampling

command_HN = 'WANDB_MODE=disabled accelerate launch \
-m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path %s \
--input_file %s \
--output_file %s \
--range_for_sampling 5-50 \
--negative_number 5 ' % (bge_model_path,align_json_file,align_json_file.replace('-align','-align-HN'))

command_align = 'WANDB_MODE=disabled accelerate launch \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir temp_data/align/%s \
--model_name_or_path %s \
--train_data %s \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 128 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" \
--save_steps 5000' % (align_pair_name,bge_model_path,align_json_file.replace('-align','-align-HN'))

align_model_path = 'temp_data/align/%s' % align_pair_name
CL_model_path = 'temp_data/align/%s-output' % align_pair_name

command_S = 'python SBert-Encode-Common.py \
--model_path %s \
--ldict_path %s \
--rdict_path %s \
--similarity_path temp_data/sim/sim_blocking_init_%s_S.npy' % (bge_model_path,
                                                             dict_ltable_path_S,
                                                             dict_rtable_path_S,
                                                             align_pair_name)

command_T = 'python SBert-Encode-Common.py \
--model_path %s \
--ldict_path %s \
--rdict_path %s \
--similarity_path temp_data/sim/sim_blocking_init_%s_T.npy' % (bge_model_path,
                                                             dict_ltable_path_T,
                                                             dict_rtable_path_T,
                                                             align_pair_name)

subprocess.run(command_HN, shell=True, capture_output=True, text=True)
subprocess.run(command_align, shell=True, capture_output=True, text=True)
result = subprocess.run(command_S, shell=True, capture_output=True, text=True)
# print(result.stdout)
result = subprocess.run(command_T, shell=True, capture_output=True, text=True)
# print(result.stdout)
similarity_S = np.load('temp_data/sim/sim_blocking_init_%s_S.npy' % align_pair_name)
similarity_T = np.load('temp_data/sim/sim_blocking_init_%s_T.npy' % align_pair_name)


### Pseudo-label in target-domain
similarity_argsort_T = (-similarity_T).argsort()
for k in [1]:
    Blocking_Result = []
    for i in range(similarity_T.shape[0]):
        for topk in range(k):
            Blocking_Result.append([i,similarity_argsort_T[i,topk],1])
    Blocking_Result_pd = pd.DataFrame(Blocking_Result).drop_duplicates()
    Blocking_Result_pd.columns = all_value_pos_T.columns
Blocking_Result_pd['sim'] = np.max(similarity_T, axis=1).tolist()
target_domain_pos = Blocking_Result_pd[Blocking_Result_pd['sim']>pseudo_label_thres]


### Cross-Domain CL Annotation

import json
# all_value = pd.concat([train,valid,test]).reset_index(drop=True)
ltable_set = all_value_S['ltable_id'].unique()
rtable_set = all_value_S['rtable_id'].unique()


with open(CL_file, 'w', encoding='utf-8') as file:
    for l_entity in tqdm(ltable_set):
        similarity_rank = (-similarity_S[l_entity]).argsort() ## 对l_entity，r_entity的相似度排序，从初始模型排序

        ltable_select = all_value_S[all_value_S['ltable_id']==l_entity] ## matching_pary中ltable_id是l_entity的dataframe
        if(ltable_select['label'].sum()>0 and ltable_select['label'].sum() < len(ltable_select)): ## 正负样本均包含
            l_entity_pos = ltable_select[ltable_select['label']==1]['rtable_id'].to_list()
            l_entity_neg = ltable_select[ltable_select['label']==0]['rtable_id'].to_list()
        elif(ltable_select['label'].sum()==0): ## 无正样本，需要采样一个正样本
            l_entity_neg = ltable_select[ltable_select['label']==0]['rtable_id'].to_list()
            
            similarity_rank = [s for s in similarity_rank if s not in l_entity_neg] ## 排除已被标注的负样本
            l_entity_pos = similarity_rank[:2]
        elif((len(ltable_select) - ltable_select['label'].sum())==0): ## 无负样本
            l_entity_pos = ltable_select[ltable_select['label']==1]['rtable_id'].to_list()
            similarity_rank = [s for s in similarity_rank if s not in l_entity_pos] ## 排除正样本
            l_entity_neg = similarity_rank[:10]
            
        # similarity_sequence = [s for s in similarity_sequence if s in rtable_set]
            # similarity_sequence = [rtable_set[s] for s in similarity_sequence]
        # l_entity_pos 是positive sample，l_entity_neg是negative sample
        ## merge_dict_list是一个函数，第一个输入是tableA或者tableB的enrich data，第二个输入是正/负样本的id集合，需要是一个list，所以pos/neg也需要分别是一个List
        positive_list = merge_dict_lists(dict_rtable_S,l_entity_pos)
        # positive_list = []
        positive_list.extend([str(most_common_values(dict_rtable_S[i])) for i in l_entity_pos]) ## 选出来most_common_list
        negative_list = merge_dict_lists(dict_rtable_S,l_entity_neg)
        # negative_list = []
        negative_list.extend([str(most_common_values(dict_rtable_S[i])) for i in l_entity_neg]) ## 选出来most_common_list
        data = {
            "query": str(most_common_values(dict_ltable_S[l_entity])),
            "pos" : positive_list,
            "neg" : negative_list
            
        }
    # 将字典转换为JSON字符串，并写入文件
        file.write(json.dumps(data, ensure_ascii=False))
        file.write('\n')  # 每个JSON对象后添加换行符
    for index,row in tqdm(target_domain_pos.iterrows()):
        l_entity_T = int(row['ltable_id'])
        r_entity_T = int(row['rtable_id'])
        query = str(most_common_values(dict_ltable_T[l_entity_T]))
        positive_list = [str(most_common_values(dict_rtable_T[r_entity_T]))]
        # positive_list.extend(merge_dict_lists(dict_rtable_T,[r_entity_T]))
        similarity_rank = (-similarity_T[l_entity_T]).argsort()
        similarity_rank = [s for s in similarity_rank if s not in [r_entity_T]]
        l_entity_neg = similarity_rank[:10]
        negative_list = [str(most_common_values(dict_rtable_T[i])) for i in l_entity_neg]
        # negative_list.extend(merge_dict_lists(dict_rtable_T,l_entity_neg))
        data = {
            "query": query,
            "pos" : positive_list,
            "neg" : negative_list
            
        }
    # 将字典转换为JSON字符串，并写入文件
        file.write(json.dumps(data, ensure_ascii=False))
        file.write('\n')  # 每个JSON对象后添加换行符
print(CL_file)

command_CL = 'WANDB_MODE=disabled accelerate launch \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir %s \
--model_name_or_path %s \
--train_data %s \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 128 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" \
--save_steps 5000' % (CL_model_path,
                      bge_model_path,
                      CL_file)
print(command_CL)
subprocess.run(command_CL, shell=True, capture_output=True, text=True)

### Evaluate

model = FlagModel(CL_model_path,use_fp16=True) ## 读取上一步的结果
embedding_a_T = model.encode([str(most_common_values(dict_ltable_T[i])) for i in range(len(dict_ltable_T))])
embedding_b_T = model.encode([str(most_common_values(dict_rtable_T[i])) for i in range(len(dict_rtable_T))])
similarity_T_output = matrix_multiply_gpu(embedding_a_T , embedding_b_T)


ltable_set = all_value_T['ltable_id'].unique()
rtable_set = all_value_T['rtable_id'].unique()
similarity_argsort = (-similarity_T_output).argsort()
recall_precision = []
for k in range(1,21,1):
# for k in [1]:
    Blocking_Result = []
    for i in range(similarity_T_output.shape[0]):
        for topk in range(k):
            Blocking_Result.append([i,similarity_argsort[i,topk],1])
    Blocking_Result_pd = pd.DataFrame(Blocking_Result).drop_duplicates()
    Blocking_Result_pd.columns = all_value_pos_T.columns
    ind_gt,ind_blocking = compare_dataframes(all_value_pos_T,Blocking_Result_pd)
    print('Top-%d Recall:%.4f Precision:%.4f' % (k,len(all_value_pos_T.loc[ind_gt]) / len(all_value_pos_T),len(Blocking_Result_pd.loc[ind_blocking]) / len(Blocking_Result_pd)))
    recall_precision.append([len(all_value_pos_T.loc[ind_gt]) / len(all_value_pos_T),len(Blocking_Result_pd.loc[ind_blocking]) / len(Blocking_Result_pd),k, len(Blocking_Result_pd)/(len(tableA_dict_T) * len(tableB_dict_T))])
result = pd.DataFrame(recall_precision)
result.columns = ['recall','precision','Top-K','CSSR']
result.to_csv('evaluation/result/%s.csv' % align_pair_name)
np.save('evaluation/result/%s.npy' % align_pair_name,similarity_T_output)

