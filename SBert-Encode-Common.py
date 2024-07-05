import numpy as np
import pandas as pd 
from tqdm import tqdm
from FlagEmbedding import FlagModel
import argparse
import torch
parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")
parser.add_argument("--model_path", type=str, default='bge-large-en-1.5', help="Model path, default is empty")
parser.add_argument("--ldict_path", type=str, default='', help="dict_ltable path")
parser.add_argument("--rdict_path", type=str, default='', help="dict_rtable path")
parser.add_argument("--similarity_path", type=str, default='', help="similarity matrix save path")
parser.add_argument("-wdc", "--wdc", action="store_true", help="Enable verbose mode")
args = parser.parse_args()
import random

def select_random_from_top_elements(lst):
    if len(lst) < 2:
        return "Error: The list does not have enough elements."
    
    # 计算列表中每个元素的长度，并创建一个包含元素及其长度的元组列表
    elements_with_lengths = [(item, len(item)) for item in lst]
    
    # 根据元素的长度降序排序
    elements_with_lengths.sort(key=lambda x: x[1], reverse=True)
    
    # 选择长度最长的前n-1个元素
    top_elements = elements_with_lengths[:len(lst)-1]
    
    # 从这些元素中随机选择一个元素
    selected_element = random.choice(top_elements)[0]
    
    return selected_element

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
try:
    dict_ltable = np.load(args.ldict_path,allow_pickle=True).item()
    dict_rtable = np.load(args.rdict_path,allow_pickle=True).item()
except:
    dict_ltable = np.load(args.ldict_path,allow_pickle=True)
    dict_rtable = np.load(args.rdict_path,allow_pickle=True)
model = FlagModel(args.model_path, 
                  use_fp16=True)
if(args.wdc):
    embedding_a = model.encode([str(select_random_from_top_elements(dict_ltable[i])) for i in range(len(dict_ltable))])
    embedding_b = model.encode([str(select_random_from_top_elements(dict_rtable[i])) for i in range(len(dict_rtable))])
else:
    embedding_a = model.encode([str(most_common_values(dict_ltable[i])) for i in range(len(dict_ltable))])
    embedding_b = model.encode([str(most_common_values(dict_rtable[i])) for i in range(len(dict_rtable))])
similarity = matrix_multiply_gpu(embedding_a , embedding_b)
np.save(args.similarity_path,similarity)