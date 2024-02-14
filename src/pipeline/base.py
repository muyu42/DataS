import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datasets import Dataset, load_dataset


def load_json_or_jsonl(data_path: str) -> None:
    #判断路径格式
    pass
    #加载json
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]    
    return data

#logger = logging.getLogger(__name__)

class BasePipeline:
    
    def __init__(self, name: str, data_path: str, output_path: str,**kwargs) -> None:
        self.name = name
        self.data_path = data_path    
        self.output_path = output_path # TODO 后续可以改成一个父目录，然后下面几个子目录存放不同文件
        self.data_format = "alpaca"   # only alpaca yet；对话形式的要再等等
        self.score_type = kwargs.get("score_type")

        if not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
    def _load_data(self, data_path: str) -> None:
        """
        Load data from data_path.
        data_path: str - path to json data file.
        默认用dataset读取，获得格式类似于pandas的
        """
        # 读取JSON文件并加载为datasets.Dataset对象
        data = load_dataset('json', data_files=data_path)['train']
        # TODO 索引 范围
        pass
        # start_idx = args.start_idx
        # end_idx = args.end_idx if args.end_idx != -1 else len(raw_datasets)
        # raw_datasets = raw_datasets.select(range(start_idx, end_idx))
        # print("load dataset from_ "+str(args.start_idx)+" _to_ "+str(args.end_idx)+" _Total = "+ str(len(raw_datasets)))
        return data
    
    def _load_other_data(self, other_data_path: str) -> None:
        raise NotImplementedError
        
    def _save_data(self, json_data) -> None:
        json_data.to_json(self.output_path, indent=4)#, ensure_ascii=False
        print(f"Saved results to {self.output_path}.")

    def _preprocess(self, json_data, other_data) -> None:
        #raise NotImplementedError
        if self.data_format == "alpaca":
            preprocessed_data = json_data
        else:
            raise ValueError(f"Data format {self.data_format} not supported.")   
        return preprocessed_data
    
    def _forward(self, preprocessed_data) -> None:
        raise NotImplementedError
    
    def run(self) -> None:
        '''
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
        json_data = self._load_data(self.data_path)
        
        other_data = None
        if hasattr(self, "other_data_path"):
            other_data = self._load_other_data(self.other_data_path)
        
        preprocessed_data = self._preprocess(json_data, other_data)
        results = self._forward(preprocessed_data)
        self._save_data(json_data, results)
        #logger.info(f"Pipeline {self.name} run complete.")
        print(f"Pipeline {self.name} run complete.")
        
class PipelineRegistry:
    
    registry = {}
    
    @classmethod
    def register(cls, name: str, pipline_class: Callable):
        
        if name in cls.registry:
            raise ValueError(f"Pipeline {name} already registered.")
        cls.registry[name] = pipline_class
    
    @classmethod
    def get_pipeline(cls, name: str):
        
        if name not in cls.registry:
            raise ValueError(f"Pipeline {name} not registered.")
        return cls.registry[name]
    
