import os
import time
import json
from pipeline.base import BasePipeline
from selection.scorer import Llama_Scorer, Mistral_Scorer
from typing import Any, Dict, List, Optional, Tuple, Union
from pipeline.utils import load_data
import logging
from tqdm import tqdm
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
class ScorePipeline(BasePipeline):#继承了BasePipeline，有不懂的看一下BasePipeline
    def __init__(self, name: str, data_path: str, **kwargs) -> None:        
        super(ScorePipeline, self).__init__(name, data_path, **kwargs) 
        self.score_func_name = "lenth"   #完成直接方法名字的定义
        #加载模型 等事情只做一次的话放在这里
        #实例化的时候运行
    def _function2score(self, example) : #可以想象成对单条qa进行打分。
        #example是个字典，里面有instruction input output
        example["lenth"] = len(example["output"])#instruction input output
        return example
    def _forward(self, preprocessed_data) -> List:
        start_time = time.time()  # 记录开始时间
        preprocessed_data.map(self._function2score)
        print('Time Used:', (time.time() - start_time) / 60, '(min)','New data len:', len(scored_dataset))        
        return preprocessed_data
    



