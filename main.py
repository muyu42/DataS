
### import
import os
import yaml
import torch
import argparse

#import sys
#sys.path.append("/data/home/lxy/dataS/evol_schedules")

from evol_schedules import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, KCenterSampling # 采样策略导入。策略是在第三方库基础上改写的
from utils import get_tokenizer, smart_tokenizer_and_embedding_resize, get_model# 加载tokenizer model  

from datasets import Dataset, load_dataset
import json
import time
from tqdm.auto import tqdm
from  analysis.data_analysis import get_perplexity_and_embedding_whole_text,get_perplexity_and_embedding_part_text
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans
from sklearn.manifold import TSNE



#TODO 数据管理，采样都写成一个类
'''
cd /data/home/lxy/dataS ; 

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --nproc_per_node=4     # 分布式  占用四张卡
    train.py  #运行train文件
    --config_file {YOUR-CONFIG-FILE} --wandb_key {YOUR-WANDB-KEY}#命令行参数，传入yaml配置文件 以及wanbd可视化的key
     > {YOUR-LOG-FILE} 2>&1 &     #打印日志到YOUR-LOG-FILE

输出文件格式
evol_res
└── {YOUR-RESULT-DIR-NAME}
    ├── data  # training data pool and unselected data pool for each iteration
    │   ├── rd_0_labeled.json 
    │   ├── rd_0_unlabeled.json
    │   ├── rd_1_labeled.json
    │   ├── rd_1_unlabeled.json
    │   ├── ...
    │   ├── rd_N_labeled.json
    │   └── rd_N_unlabeled.json
    └── output  # instruction-tuned chat model for each iteration
        ├── rd_0
        ├── rd_1
        ├── ...
        └── rd_N 

'''

## GET_EVOL_SCHEDULES   选择策略
def get_evol_schedule(evol_schedule_name):
    if evol_schedule_name == "RandomSampling":
        return RandomSampling
    if evol_schedule_name == "LeastConfidence":
        return LeastConfidence
    elif evol_schedule_name == "MarginSampling":
        return MarginSampling
    elif evol_schedule_name == "EntropySampling":
        return EntropySampling
    elif evol_schedule_name == "KCenterSampling":
        return KCenterSampling
    # elif evol_schedule_name == "VendiSampling":
    #     return VendiSampling


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
############################################################################################
## args
class Flag:
    pass
if 1:
    model_name_or_path = "/data/LLMs/shakechen/Llama-2-7b-chat-hf/"
    output_dir = "./data"


    args = Flag()
    args.path = ""
    args.model_max_length = 512
    args.data_path = "/data/home/lxy/taskV/use_yi2generate/rawdata/alpaca_data.json"
    args.save_path = "/data/home/lxy/dataS/data/datapre/alpaca_data_pre.pt"
    args.json_save_path = "/data/home/lxy/dataS/data/select_rd_0/alpaca_data_select.json"

    args.start_idx = 0
    args.end_idx = 20
    #args.mod4template = "ins-inp-outp"#args.need_output_i = 1
    args.max_length = 512

    #mod = "pre"  启动analysis流程，获取ppl和embeding
    args.mod = "pre"

    args.cluster_method='kmeans'
    args.reduce_method='tsne'
    args.sample_num =5 
    args.kmeans_num_clusters =3 
    args.low_th =25 
    args.up_th =75





    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

if 1:
    initflag = 0
    analysisflag  = 0

############################################################################################
### init
print("init..",initflag)
if initflag:
    model = get_model(model_name_or_path=model_name_or_path, device_map="auto",output_hidden_states=True)#output_hidden_states可以改成依据需不要中间值而改变

    model.eval()

    tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=model_name_or_path,model_max_length=args.model_max_length,)

    tokenizer, model = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                                        tokenizer=tokenizer, 
                                                                        model=model)  # fix tokenizer's special_token_maps

print("init_finished")
#exit()
############################################################################################
### data 
##QA
#TODO 重构数据，给qa打标签？  迎合那个主动学习框架


#TODO 目前是读取单个文件，可以改成文件夹
#data_files = {"train": args["data_path"]}
#raw_datasets = load_dataset('json', data_files=data_files)
# 读取JSON文件并加载为datasets.Dataset对象
raw_datasets = load_dataset('json', data_files=args.data_path)['train']

# 设置要处理的数据范围
start_idx = args.start_idx
end_idx = args.end_idx if args.end_idx != -1 else len(raw_datasets)
raw_datasets = raw_datasets.select(range(start_idx, end_idx))
print("load dataset from_ "+str(args.start_idx)+" _to_ "+str(args.end_idx)+" _Total = "+ str(len(raw_datasets)))
#print(raw_datasets[1])

############################################################################################
### get embeding/ppl -> pkl  use llama
print("starting analysis by llama...")
if analysisflag:
    
    if args.save_path[-3:] != '.pt':
        args.save_path += '.pt'
    if os.path.exists(args.save_path):
        print('查看是否已经有了，有了就报错-save_path exists!')
        raise Exception


    #TODO 参考/data/home/lxy/taskV/use_yi2generate/select_by_IFD/data_analysis.py和llamafactory，
        #通过模板，构造输入数据。
    def trans_temp2use( data_i,PROMPT_DICT,mod4template="ins-inp-outp"):
        '''
        data_i 包含 instruction input output 三个字段    后续可以扩展
        mod4template = "ins-inp-outp" 用户控制识别模式。在聚类的时候不用output

        prompt = 'alpaca'  用来选择数据是alpaca  还是wiz的prompt  具体可以参考llama factory

        预期的输出
        promt_to_use = ins+inp:
        whole_text = ins+inp+outp
        (only ins = ins
        (direct_answer_text = '### Response:' + outp

        '''
        #
        instruct_i = data_i['instruction'] if 'instruction' in data_i.keys() else ''
        output_i = data_i['output']  if 'output' in data_i.keys() else '' 
        input_i = data_i['input'] if 'input' in data_i.keys() else ''
        #data_i['output']   # 我去掉了output的影响   我记得tegit说要解耦input，感觉input拆开也可以

        # 根据mod字符串进行解码，
        #print("not support wiz yet")
        if input_i == '':
            temp_dict = {'instruction':instruct_i}
            promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        else:
            temp_dict = {'instruction':instruct_i,'input':input_i}
            promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)

        whole_text = promt_to_use + output_i

        if "outp" in mod4template:
            return whole_text
        else:

            return promt_to_use

        




    # 准备处理数据的功能函数
    def process_data(example, tokenizer, model, mod,PROMPT_DICT):
        # 1 构造文本和其他处理，和之前保持一致
        promt_to_use = trans_temp2use( example,PROMPT_DICT,mod4template="ins-inp")
        #direct_answer_text = trans_temp2use( example,PROMPT_DICT,mod4template="outp")
        whole_text =trans_temp2use( example,PROMPT_DICT,mod4template="ins-inp-outp")
        output_i = example['output']#trans_temp2use( example,PROMPT_DICT,mod4template="ins-inp-outp")
        direct_answer_text = '### Response:' + output_i

        # 2根据 args.mod 参数进行处理
        if mod == 'pre':
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, promt_to_use, args.max_length)
            processed_example = { "ppl_ins_alone":ppl_ins_alone,"emb_ins_alone":emb_ins_alone}  # 这里只是示例，需要替换为实际的数据处理逻辑
            #"ppl": [ppl_ins_alone, 0, 0], 
        elif mod == 'cherry':
            # 对于 cherry 模式的处理，同样你需要确保相应的函数已经定义
            instruct_i_input_ids = tokenizer.encode(promt_to_use, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            instruct_i_len = instruct_i_input_ids.shape[1] 
        
            ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text, output_i, args.max_length-instruct_i_len+4)
            ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)

    
            processed_example = { "ppl": [0,ppl_out_alone,ppl_out_condition], "token_loss":[[],loss_list_alone,loss_list_condition] }     
        else :
            processed_example = {"ppl":'cherry'}

        # 你可以处理并返回数据结构中的数据
        #print("promt_to_use",promt_to_use)
        #print("processed_example",processed_example)
        return processed_example

    # 设置部分参数
    process_data_partial = partial(process_data, tokenizer=tokenizer, model=model, mod='pre',PROMPT_DICT=PROMPT_DICT)

    # 现在可以作为函数传递给datasets的map方法# 使用 apply 方法对数据集中的每个样本调用 process_data 函数

    start_time = time.time()  # 记录开始时间
    processed_dataset = raw_datasets.map(process_data_partial) # output_file将处理后的数据保存到指定的 JSON 文件中)#, batched=False表明逐个处理数据项,remove_columns=["instruction"],remove_columns忽略 "instruction" 列
    print('Time Used:', (time.time() - start_time) / 60, '(min)')
    #processed_data = processed_dataset.to_dict()  # 转换成字典
    #print('New data len:', len(processed_data))


    processed_dataset.to_json(args.save_path, force_ascii=False)
    #
    # processed_data_list = processed_dataset[:]
    # torch.save(processed_data_list,args.save_path)

    # # 保存处理后的数据，这里保存为JSON格式
    # with open(args.save_path, 'w') as f_out:
    #     json.dump(processed_data, f_out)


############################################################################################
### score
##QAS  (S是一个json)

def function2score(example,name = "yiapi"):
    '''
    example 是个字典，包含 instruction input output 三个key    （后续可以扩展
    name = "yiapi"  用户控制识别模式。在聚类的时候不用output

    prompt = 'alpaca'  用来选择数据是alpaca  还是wiz的prompt  具体可以参考llama factory

    预期的输出
    promt_to_use = ins+inp:
    whole_text = ins+inp+outp
    (only ins = ins
    (direct_answer_text = '### Response:' + outp

    '''
    pass
    example["scores"] = {"llm":0.5,"ifd":0.6}
    return example
# 设置部分参数
function2score_partial = partial(function2score, name = "yiapi")

start_time = time.time()  # 记录开始时间
scored_dataset = raw_datasets.map(function2score_partial,)
print('Time Used:', (time.time() - start_time) / 60, '(min)','New data len:', len(scored_dataset))



#exit()
############################################################################################
### 聚类
############################################################################################
### 聚类
############################################################################################
### 聚类
##QAS,C

if not analysisflag:
    #加载之前保存的
    processed_dataset = load_dataset('json', data_files=args.save_path)['train']

def do_clustering(args, high_dim_vectors):
    #可以补充 后续使用其他聚类方法
    print()
    clustering_algorithm = args.cluster_method
    if clustering_algorithm == 'kmeans':
        clustering = KMeans(n_clusters=args.kmeans_num_clusters, random_state=0,n_init='auto').fit(high_dim_vectors)
    return clustering

#pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
print ("导入data analysis里面保存的 ppl 和embeding")
pt_data = processed_dataset




emb_list = []
ppl_list = []

print("len(pt_data)",len(processed_dataset))
## 获取每个数据的embedding，用于kmeans聚类别；
## 获取每个数据的ppl值，也就是loss
#     
#代码通过遍历`pt_data`的长度，从每个数据项中提取`sent_emb`和`ppl`属性的值，
# 并将其添加到`emb_list`和`ppl_list`中
for i in tqdm(range(len(pt_data))):
    data_i = pt_data[i]
    #print("data_i",data_i)
    sent_emb_list = torch.Tensor(data_i['emb_ins_alone'])
    emb_list.append(sent_emb_list)
    ppl_list.append(data_i['ppl_ins_alone'])
    #print(len(data_i['emb_ins_alone']),data_i['emb_ins_alone'])

high_dim_vectors = torch.cat(emb_list,0).numpy()
#通过将`emb_list`转换为NumPy数组，创建了一个名为`high_dim_vectors`的高维向量，每一个qa对应的embed为 4096.high_dim_vectors维度 [num_QA, 4096]
ppl_array = np.array(ppl_list)

print("## 使用kmeans进行聚类")#对`high_dim_vectors`进行聚类，得到一个聚类结果
clustering = do_clustering(args, high_dim_vectors)
cluster_labels = clustering.labels_
# 通过`clustering`获取每个数据点的聚类标签，保存在`cluster_labels`变量中

#TODO cluster_labels 保存到json里面，注意起末index对应关系

# Perform t-SNE for visualization.TODO 注意参数设置n_components，perplexity，
if args.reduce_method == 'tsne':
    print("draw_tsne")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)  # PCA到50维
    X_pca = pca.fit_transform(high_dim_vectors)

    # 使用 t-SNE 进行可视化
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000, random_state=0)
    low_dim_vectors = tsne.fit_transform(X_pca)

    # 可视化
    plt.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=cluster_labels, s=5, cmap='Spectral')
    plt.title('Improved t-SNE Visualization of Clusters')
    plt.colorbar(ticks=range(max(cluster_labels)+1))  # 根据类别数设置colorbar的ticks
    plt.clim(-0.5, max(cluster_labels) + 0.5)  # 根据类别数设置colorbar的范围
    plt.savefig('improved_tsne_visualization3.png')  # 保存图片
    plt.show()    

print("完成聚类")


num_clusters = len(np.unique(cluster_labels))#首先计算出数据中的聚类个数
#对于一维数组或者列表，np.unique() 函数 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表
print("num_clusters",num_clusters)

# Get the indices for each cluster
#对于每个聚类，找出属于该聚类的样本的索引，存储在cluster_indices/samples字典中  "i": np.where(cluster_labels == i)[0]  
#某一聚类样本  对应的源文件的索引
cluster_indices = {i: np.where(cluster_labels == i)[0] for i in range(num_clusters)}
print("debug",cluster_indices)

############################################################################################
### sample  是一个middle_confidence_samples = {} 字典 存储着    某一聚类样本  对应的源文件的索引
##QAS,C,label1


#从数据中取出中等置信水平的样本。函数接受以下参数
# cluster_labels：数据点的聚类标签
# confidences：样本的置信水平
# n：要取出的样本数量
# low_th：置信水平下界，默认值为25
# up_th：置信水平上界，默认值为75
def sample_middle_confidence_data(cluster_indices, confidences, n, low_th=25, up_th=75):
    #TODO check cluster_indices是否符合规范
    num_clusters = len(cluster_indices.keys())#看看是否和args的一样
    
    #接下来，函数创建一个空的字典middle_confidence_samples，用来存储中等置信水平的样本的索引
    middle_confidence_samples = {}

    for i in range(num_clusters):#对于每个聚类，函数按以下步骤处理
        # Get the sorted indices for this cluster
        #获取属于该聚类的样本的索引，并对这些样本的置信水平进行排序
        sorted_indices = cluster_indices[i]
        
        #如果该聚类的样本数量小于要取出的样本数量n，
        # 则直接将所有的样本索引存储在middle_confidence_samples中，
        # 并继续处理下一个聚类
        if len(sorted_indices) < n:
            middle_confidence_samples[i] = sorted_indices
            print("not enough len(sorted_indices) < n",len(sorted_indices),n)
            continue

        #计算该聚类的置信水平的下界和上界，并找出处于中等置信水平范围内的样本的索引
        cluster_confidences = confidences[sorted_indices]
        lower_threshold = np.percentile(cluster_confidences, low_th)
        upper_threshold = np.percentile(cluster_confidences, up_th)

        # Get the indices of the samples within the middle level confidence range
        #并找出处于中等置信水平范围内的样本的索引
        middle_indices = sorted_indices[(cluster_confidences >= lower_threshold) & (cluster_confidences <= upper_threshold)]
        
        # If there are less than n samples in the middle range, use all of them
        if len(middle_indices) < n:
            #若中等置信水平范围内的样本数量小于要取出的样本数量n，
            # 则直接将所有的这些样本索引存储在middle_confidence_samples中
            middle_confidence_samples[i] = middle_indices
            print("not enough len(middle_indices) < n",len(middle_indices),n)
        else:
            #若中等置信水平范围内的样本数量满足（大于等于）要取出的样本数量n， 则等间隔选取n个点
            # Calculate step size for even sampling
            step_size = len(middle_indices) // n
            middle_confidence_samples[i] = middle_indices[::step_size][:n]

    return middle_confidence_samples


print("## 获取中间置信度的数据")


print("#获得索引")
middle_confidence_samples = sample_middle_confidence_data(cluster_indices, ppl_array, args.sample_num, args.low_th, args.up_th)



############################################################################################
#根据索引获得新的数据
#TODO 注意范围
#raw_datasets是要筛选的数据 在之前就加载过了    注意范围   args.start_idx

print("#根据索引获得新的数据")
#TODO 重写         采用字典的格式，对应回去，，，，，还是采样datasets呢？   遵循原来的顺序吗  还是聚类排序之后的顺序
def get_json_sample(middle_confidence_samples):#从聚类标签和条件信息中获取JSON样本
    json_samples = []
    for k in middle_confidence_samples.keys():
        ids_list = middle_confidence_samples[k].tolist()
        for id_i in ids_list:
            ori_sample = raw_datasets[id_i]
            json_samples.append(ori_sample)

    return json_samples

new_data = get_json_sample(middle_confidence_samples)
print('New data len \n',len(new_data))#看一下 new_data是什么格式 json list


dataset = Dataset.from_dict(new_data)
# 将JSON对象列表转换为pandas的DataFrame
new_datadf = pd.DataFrame(new_data)
# 将pandas的DataFrame转换为datasets格式
dataset = Dataset.from_pandas(new_datadf)


dataset.to_json(args.json_save_path, indent=4)




'''
cd /data/home/lxy/dataS ; 

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --nproc_per_node=4     # 分布式  占用四张卡
    train.py  #运行train文件
    --config_file {YOUR-CONFIG-FILE} --wandb_key {YOUR-WANDB-KEY}#命令行参数，传入yaml配置文件 以及wanbd可视化的key
     > {YOUR-LOG-FILE} 2>&1 &     #打印日志到YOUR-LOG-FILE

输出文件格式
evol_res
└── {YOUR-RESULT-DIR-NAME}
    ├── data  # training data pool and unselected data pool for each iteration
    │   ├── rd_0_labeled.json 
    │   ├── rd_0_unlabeled.json
    │   ├── rd_1_labeled.json
    │   ├── rd_1_unlabeled.json
    │   ├── ...
    │   ├── rd_N_labeled.json
    │   └── rd_N_unlabeled.json
    └── output  # instruction-tuned chat model for each iteration
        ├── rd_0
        ├── rd_1
        ├── ...
        └── rd_N 

'''
############################################################################################
## RUN
# def main(config_file):

#     print("DiverseEvol Done ^_^")


# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--config_file', type=str, required=True,)
#     # parser.add_argument('--wandb_key', type=str, default='b189be602734f52ac19168f0656370c1bd309771')
#     # args = parser.parse_args()
    
#     # import wandb
#     # wandb.login(key=args.wandb_key)
    
#     main()#config_file=args.config_file
