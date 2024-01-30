
import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
import time
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans
from sklearn.manifold import TSNE
print(os.environ["CUDA_VISIBLE_DEVICES"])



############################################################################################
## args
class Flag:
    pass
if 1:
    model_name_or_path = "/data/LLMs/01ai/Yi-6B-Chat"
    #output_dir = "./data"
    args = Flag()
    args.path = ""
    args.model_max_length = 512
    args.data_path = "自己路径/data/rawdata/alpaca_data.json"
    args.json_save_path = "自己路径/data/alpaca_data_select.json"

    args.start_idx = 0
    args.end_idx = 20

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


############################################################################################
### init

pass
print("init_finished")
#exit()
############################################################################################
### data 
##QA

# 读取JSON文件并加载为datasets.Dataset对象
raw_datasets = load_dataset('json', data_files=args.data_path)['train']

# 设置要处理的数据范围
start_idx = args.start_idx
end_idx = args.end_idx if args.end_idx != -1 else len(raw_datasets)
raw_datasets = raw_datasets.select(range(start_idx, end_idx))
print("load dataset from_ "+str(args.start_idx)+" _to_ "+str(args.end_idx)+" _Total = "+ str(len(raw_datasets)))
#print(raw_datasets[1])

############################################################################################
### score
##QAS  

def function2score(example,name = "lenthofins"):
    '''
    example 是个字典，包含 instruction input output 三个key   
    name = "lenthofins"  用户控制识别模式。，作为这个方法的正式名称，之后管理score方法的时候用

    '''
    score_lenth = len(example["instruction"])#不用考虑归一化，后续会有步骤进行norm

    example["score_lenth"] = score_lenth
    return example

# 设置部分参数
function2score_partial = partial(function2score, name = "lenthofins")

start_time = time.time()  # 记录开始时间
scored_dataset = raw_datasets.map(function2score_partial,)
print('Time Used:', (time.time() - start_time) / 60, '(min)','New data len:', len(scored_dataset))


scored_dataset.to_json(args.json_save_path, indent=4)



# if __name__ == '__main__':
