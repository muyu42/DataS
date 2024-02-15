'''
舍弃
'''

import torch
import copy
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

## Model conversion utils
#在模型参数字典和一维张量之间进行转换
def state_dict_to_vector(state_dict, remove_keys=[]):
    '''
    首先使用deepcopy函数复制输入的state_dict，并将复制结果存储在shared_state_dict中。
    依次遍历remove_keys列表中的键，如果键存在于shared_state_dict中，则删除该键。
    使用sorted函数对shared_state_dict按键进行排序，并将排序结果存储在sorted_shared_state_dict中。
    
    遍历sorted_shared_state_dict中的每一项，将值进行形状重塑为一维张量，并存储在列表中。
    调用torch.nn.utils.parameters_to_vector函数，将列表中的一维张量转换为一个完整的一维张量，并将结果返回。
    '''
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    '''
    首先使用deepcopy函数复制输入的state_dict，并将复制结果存储在reference_dict中。
    依次遍历remove_keys列表中的键，如果键存在于reference_dict中，则删除该键。
    使用sorted函数对reference_dict按键进行排序，并将排序结果存储在sorted_reference_dict中。
    
    调用torch.nn.utils.vector_to_parameters函数，将输入的一维张量按照sorted_reference_dict的顺序赋值给模型参数。
    如果"transformer.shared.weight"存在于sorted_reference_dict中，则将remove_keys列表中的键的值设置为sorted_reference_dict["transformer.shared.weight"]的值。
    返回sorted_reference_dict作为转换后的模型参数字典。
    '''
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

class TaskVector():
    '''
        这段代码定义了一个名为 TaskVector 的类，用于处理任务向量相关的操作。下面是对每个方法的解释和思路流程：

        __init__ 方法
        该方法用于初始化任务向量对象，可以通过预训练的检查点和微调后的检查点来创建任务向量，或者直接传入任务向量的状态字典。
        如果传入了任务向量状态字典，则直接使用该状态字典作为任务向量；否则，从预训练和微调的状态字典中计算任务向量。

        __add__ 方法
        该方法用于将两个任务向量相加，返回一个新的任务向量对象。
        遍历每个键，将两个任务向量对应键的值相加，得到新的任务向量。

        __radd__ 方法
        该方法定义了反向相加操作，当另一个对象为整数或空时，返回本身；否则调用__add__ 方法实现加法操作。

        __neg__ 方法
        该方法用于对任务向量进行取负操作，返回一个新的任务向量对象。
        遍历每个键，将任务向量的值取负，得到新的任务向量。

        weightmerging 方法
        该方法用于对多个任务向量进行加权合并，返回一个新的任务向量对象。
        遍历每个键，对多个任务向量对应键的加权和进行计算，得到新的任务向量。

        apply_to 方法
        该方法用于将任务向量应用到预训练模型中，返回应用了任务向量的新模型。
        遍历每个键，对预训练模型的状态字典进行相应的更新，得到新的模型。
        整体上，这个 TaskVector 类拥有一系列对任务向量进行操作的方法，能够方便地进行任务向量的组合、应用和转换等操作
        
    '''
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """

        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
import torch

class TaskVector:
    """
    这段代码定义了一个名为 TaskVector 的类，用于处理任务向量相关的操作。下面是对每个方法的解释和思路流程：
    
    __init__ 方法
    该方法用于初始化任务向量对象，可以通过预训练的检查点和微调后的检查点来创建任务向量，或者直接传入任务向量的状态字典。
    如果传入了任务向量状态字典，则直接使用该状态字典作为任务向量；否则，从预训练和微调的状态字典中计算任务向量。
    
    __add__ 方法
    该方法用于将两个任务向量相加，返回一个新的任务向量对象。
    遍历每个键，将两个任务向量对应键的值相加，得到新的任务向量。
    
    __radd__ 方法
    该方法定义了反向相加操作，当另一个对象为整数或空时，返回本身；否则调用__add__ 方法实现加法操作。
    
    __neg__ 方法
    该方法用于对任务向量进行取负操作，返回一个新的任务向量对象。
    遍历每个键，将任务向量的值取负，得到新的任务向量。
    
    weightmerging 方法
    该方法用于对多个任务向量进行加权合并，返回一个新的任务向量对象。
    遍历每个键，对多个任务向量对应键的加权和进行计算，得到新的任务向量。
    
    apply_to 方法
    该方法用于将任务向量应用到预训练模型中，返回应用了任务向量的新模型。
    遍历每个键，对预训练模型的状态字典进行相应的更新，得到新的模型。
    
    整体上，这个 TaskVector 类拥有一系列对任务向量进行操作的方法，能够方便地进行任务向量的组合、应用和转换等操作。
    """

    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """
        Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None, "Pretrained and finetuned checkpoints must be provided."
            pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))
            finetuned_state_dict = torch.load(finetuned_checkpoint, map_location=torch.device('cpu'))
            self.vector = {key: finetuned_state_dict[key] - pretrained_state_dict[key] for key in pretrained_state_dict if pretrained_state_dict[key].dtype not in [torch.int64, torch.uint8]}

    def __add__(self, other):
        """Add two task vectors together and return a new task vector."""
        assert isinstance(other, TaskVector), "Operand should be an instance of TaskVector."
        new_vector = {key: self.vector[key] + other.vector[key] for key in self.vector if key in other.vector}
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        """Reverse add which allows summation with None or int to return self."""
        if other is None or isinstance(other, int):
            return self
        else:
            return self.__add__(other)

    def __neg__(self):
        """Negates the task vector and return a new task vector."""
        new_vector = {key: -self.vector[key] for key in self.vector}
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        """Merge multiple task vectors according to given weights and return a new task vector."""
        assert len(taskvectors) == len(coefficients), "The number of task vectors and coefficients should be the same."
        new_vector = {key: sum(tv.vector[key] * weight for tv, weight in zip(taskvectors, coefficients)) for key in self.vector}
        return TaskVector(vector=new_vector)
    
    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply the task vector to a pretrained model and return the updated model."""
        pretrained_model_state_dict = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))
        new_state_dict = {key: pretrained_model_state_dict[key] + scaling_coef * self.vector[key] if key in self.vector else pretrained_model_state_dict[key] for key in pretrained_model_state_dict}
        return new_state_dict
    
# Usage examples can be inserted here
'''
# Assuming you have already defined the TaskVector class as provided above

# Paths to checkpoint files for pretrained and finetuned models
pretrained_checkpoint_path = 'path_to_pretrained_model_checkpoint.pth'
finetuned_checkpoint_path = 'path_to_finetuned_model_checkpoint.pth'

# Example 1: Initialize a TaskVector from checkpoints
task_vector_from_checkpoints = TaskVector(pretrained_checkpoint=pretrained_checkpoint_path, finetuned_checkpoint=finetuned_checkpoint_path)

# Example 2: Initialize a TaskVector from an existing vector state dictionary
existing_vector_state_dict = {'layer.weight': torch.tensor([1.0, 2.0, 3.0]),
                              'layer.bias': torch.tensor([-1.0, -2.0, -3.0])}
task_vector_from_vector = TaskVector(vector=existing_vector_state_dict)

# Example 3: Add two TaskVectors together
task_vector1 = TaskVector(pretrained_checkpoint=pretrained_checkpoint_path, finetuned_checkpoint=finetuned_checkpoint_path)
task_vector2 = TaskVector(pretrained_checkpoint=pretrained_checkpoint_path, finetuned_checkpoint=finetuned_checkpoint_path)
added_task_vector = task_vector1 + task_vector2

# Example 4: Negate a TaskVector
negated_task_vector = -task_vector1

# Example 5: Weighted merge of multiple TaskVectors
task_vector3 = TaskVector(pretrained_checkpoint=pretrained_checkpoint_path, finetuned_checkpoint=finetuned_checkpoint_path)
tvs = [task_vector1, task_vector2, task_vector3]
coefficients = [0.2, 0.5, 0.3]
weighted_merge_vector = task_vector1.weightmerging(tvs, coefficients)

# Example 6: Apply TaskVector to pretrained model
updated_state_dict = task_vector1.apply_to(pretrained_checkpoint_path)
model = torch.nn.Module()  # You need to replace this with the actual model class
model.load_state_dict(updated_state_dict, strict=False)  # Load the new state dict into the model

'''
class GetInfo:
    def __init__(self, model1_name, model2_name, data_path):
        # Load the models
        self.model1 = AutoModel.from_pretrained(model1_name)
        self.model2 = AutoModel.from_pretrained(model2_name)

        # Set the models to eval mode
        self.model1.eval()
        self.model2.eval()

        # Load the dataset from the jsonl file
        self.dataset = load_dataset('json', data_files=data_path)['train']

    def compute_task_vector(self):
        # Ensure both models are on the same device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_1.to(device)
        self.model_2.to(device)
        
        # Get the parameters of both models
        params_model_1 = {k: v.clone() for k, v in self.model_1.named_parameters()}
        params_model_2 = {k: v.clone() for k, v in self.model_2.named_parameters()}
        
        # Compute task vector by subtracting the parameters of model 1 from model 2
        task_vector = {name: (params_model_2[name] - params_model_1[name]).cpu() for name in params_model_1}
        return task_vector
    def calculate_task_vector(self):
        task_vector = {}

        # Get parameters from each model and calculate task vector.
        # We will assume both models contain the same parameters keys for simplicity
        params1 = self.model1.named_parameters()
        params2 = self.model2.named_parameters()
        dict_params2 = dict(params2)

        for name1, param1 in params1:
            # Only layers with learnable parameters are considered (Conv, Linear, etc.)
            if param1.requires_grad:
                # Calculate the task vector (difference of the parameters)
                task_vector[name1] = param1.data - dict_params2[name1].data

        return task_vector

    def save_task_vector(self, task_vector, save_path):
        # Initialize storage for parameters
        task_vector_storage = {}

        # Convert task_vector to a format that can be saved with torch.save
        for key, value in task_vector.items():
            task_vector_storage[key] = value.cpu().detach()

        # Save task vector in PyTorch file format
        torch.save(task_vector_storage, f"{save_path}/task_vector.pt")





        

    


# Example of usage
model1_name = 'bert-base-uncased'  # specify the first model name
model2_name = 'bert-base-uncased'  # specify the second model name with further training or modifications
# and "data_path" is the path to a jsonl file as expected by the datasets library
data_path = 'data.jsonl'  # specify the path to your JSONL file
save_path = 'model_info_output'  # specify the output directory



info = GetInfo(model1_name, model2_name, data_path)
task_vector = info.calculate_task_vector()
info.save_task_vector(task_vector, save_path)
