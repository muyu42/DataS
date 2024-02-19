import json
import os
import sys

import torch
import torch.nn as nn
import numpy as np
sys.path.append("/data/home/guanchaofeng/DataS/src")
import time
from pipeline.base import BasePipeline
from typing import Any, Dict, List
from pipeline.utils import load_data
import logging
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer,LlamaTokenizer, LlamaForCausalLM,AutoModelForCausalLM

logger = logging.getLogger(__name__)

'''
    def run(self) -> None:
        ##主要流程
        #1 加载数据，目前仅支持alpaca格式（for单论sft），对话格式之后添加
        json_data = self._load_data(self.data_path)
        #2添加 其他数据辅助，默认没有
        other_data = None
        if hasattr(self, "other_data_path"):
            other_data = self._load_other_data(self.other_data_path)
        #3预处理 需要的话写，不需要就直接传json_data过去了
        preprocessed_data = self._preprocess(json_data, other_data)
        #4 给整个数据集所有qa打分
        results = self._forward(preprocessed_data)
        #5保存
        self._save_data(json_data, results)
        logger.info(f"Pipeline {self.name} run complete.")
'''
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用编号为0的GPU
#print(torch.cuda.device_count())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class ScorePipeline(BasePipeline):#继承了BasePipeline，有不懂的看一下BasePipeline
    
    def __init__(self, name: str, data_path: str, output_path:str, **kwargs) -> None:        
        super(ScorePipeline, self).__init__(name, data_path, output_path, **kwargs) 
        
        self.score_func_name = "ifd"   #完成直接方法名字的定义
        #加载模型 等事情只做一次的话放在这里
        #实例化的时候运行
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if_cuda = torch.cuda.is_available()
        print("if_cuda=",if_cuda)
        gpu_count = torch.cuda.device_count()
        print("gpu_count=",gpu_count)

        self.model_name = '/data/LLMs/01ai/Yi-34B-Chat'
        #self.model_name = '/data/home/guanchaofeng/LLMs/01ai/Yi-6B-Chat'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.rank_model =  AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", output_hidden_states=True)
        self.rank_model.eval()

        self.log_softmax = nn.LogSoftmax(dim=-1).to(self.device)
        self.nll_loss = nn.NLLLoss(reduction='none').to(self.device)
        self.sample_rate = 0.1
        self.sample_number = 0
        
        self.PROMPT_DICT = {
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
        
    def get_perplexity_and_embedding_part_text(self,tokenizer, model, text, target_span, max_length):

        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        end_token = input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        losses = []
        logits = outputs.logits
        for i in range(1, end_token):
            log_prob_dist = self.log_softmax(logits[0, i-1])
            true_token = input_ids[0, i]
            token_loss = self.nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
            losses.append(token_loss.item())

        return perplexity.to('cpu'), 0, losses

    def get_loss_part_text(self,tokenizer, text, target_span, max_length, loss_list_):

        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
        start_index = text.rfind(target_span)
        text_temp = text[:start_index]
        token_id_temp = tokenizer.encode(text_temp)
        start_token = len(token_id_temp) 
        end_token_real = input_ids.shape[1]

        loss_list = loss_list_[start_token-1:end_token_real-1] 

        return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
    

        
    def _function2score(self, sampled_data,max_length=100) : #对多条qa进行打分。
        #sampled_data是个字典，里面有instruction input output
        pt_data = []
        mean_rate_list = []
        mean_list_1 = []
        mean_list_2 = []
        
        for i in tqdm(range(len(sampled_data))):
            data_i = sampled_data[i]
            instruct_i = data_i['instruction']
            output_i = data_i['output']

            direct_answer_text = '### Response:' + output_i

            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = self.PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = self.PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

            temp_data_i = {}

            instruct_i_input_ids = self.tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            instruct_i_len = instruct_i_input_ids.shape[1] 

            ppl_out_alone, _, loss_list_alone = self.get_perplexity_and_embedding_part_text(self.tokenizer, self.rank_model, direct_answer_text, output_i, max_length-instruct_i_len+4)
            ppl_out_condition, _, loss_list_condition = self.get_perplexity_and_embedding_part_text(self.tokenizer, self.rank_model, whole_text, output_i, max_length)

            temp_data_i['ppl'] = [0,ppl_out_alone,ppl_out_condition]
            temp_data_i['token_loss'] = [[],loss_list_alone,loss_list_condition]

            pt_data.append(temp_data_i)

        print('New data len:', len(pt_data))
        for i in tqdm(range(len(pt_data))):

            pt_data_i = pt_data[i]
            loss_1_list = pt_data_i['token_loss'][1]
            loss_2_list = pt_data_i['token_loss'][2]

            json_data_i = sampled_data[i]
            instruct_i = json_data_i['instruction']
            output_i = json_data_i['output']

            direct_answer_text = '### Response:' + output_i

            input_i = json_data_i['input']
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = self.PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = self.PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

            # Tokenize the input text
            instruct_i_input_ids = self.tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            instruct_i_len = instruct_i_input_ids.shape[1]
            
            if max_length-instruct_i_len > 0:

                len_1, token_ids_1, loss_list_1 = self.get_loss_part_text(self.tokenizer, direct_answer_text, output_i, max_length-instruct_i_len+4, loss_1_list).to(self.device)
                len_2, token_ids_2, loss_list_2 = self.get_loss_part_text(self.tokenizer, whole_text, output_i, max_length, loss_2_list).to(self.device)

                if len_1 <= 0 or len_2 <= 0:
                    continue

                if instruct_i_len + len_1 > max_length:
                    continue

                mean_1 = loss_list_1.mean()
                mean_2 = loss_list_2.mean()
                mean_rate = mean_2/mean_1

                if mean_rate > 1: 
                    continue
                
                mean_rate_list.append((mean_rate,i))
                mean_list_1.append((mean_1,i))
                mean_list_2.append((mean_2,i))
                # print('mean_rate',mean_rate)
                # print('sampled_data',sampled_data)
                sampled_data[i]['score_ifd'] = mean_rate

            else:
                continue
        print('mean_rate_list',mean_rate_list)
        print('mean_list_1',mean_list_1)
        print('mean_list_2',mean_list_2)
        print('sampled_data',sampled_data)
        return sampled_data
    
    
    def _forward(self, preprocessed_data) -> List:
        
        start_time = time.time()  # 记录开始时间
        results_data = self._function2score(preprocessed_data)
        print('Time Used:', (time.time() - start_time) / 60, '(min)')  
        return results_data 
             
        

if __name__ == '__main__':
    pipeline = ScorePipeline(name = './', 
    data_path = './',  
    output_path = './'
    )
        
    example = [
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
    {
        "instruction": "What are the three primary colors?",
        "input": "",
        "output": "The three primary colors are red, blue, and yellow."
    },
    {
        "instruction": "Describe the structure of an atom.",
        "input": "",
        "output": "An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom."
    }
    ]
    #[{'instruction': 'instruction', 'input': 'input', 'output': 'output'}]
    with open('/data/home/guanchaofeng/Score/data/output/test_data_1w.json', 'r',encoding='utf-8') as f:
        data = json.load(f)
    output = pipeline._forward(data)
    with open('/data/home/guanchaofeng/DataS/output/test_data_1w.json', 'w',encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

