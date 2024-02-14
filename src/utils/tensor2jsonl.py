
'''
存储和读取高维 Tensor 对象到 jsonl 文件，
需要序列化和反序列化 Tensors、创建一个 Dataset 对象来保存这些数据，然后将数据集写入 jsonl 文件。
再从 jsonl 文件读取时，你可以重新创建 Dataset 对象并将数据恢复为 Tensor 对象。

首先需要安装 datasets 库，可以通过以下命令行来安装：
pip install datasets


对于 0 值：tensor.tolist() 方法将会完全保留所有的元素，包括 0 值。在列表中，这些 0 值会被简单地表示为数字 0。
对于 NaN/inf 值：如果你的 Tensor 中包含这样的特殊值，tolist() 会原封不动地保留它们，并在转换为 Python 列表后以 float('nan')、float('inf')、float('-inf') 表示。
关于 梯度保留 的问题：tolist() 方法并不会保留 Tensor 的梯度信息。实际上，当你将 Tensor 转换为 Python 的标准数据类型（比如列表或数值）时，所有和自动微分（auto-grad）机制相关的信息都会丢失。.tolist() 是将数据简单地转换为 Python 原生数据类型的方法，这是一个单向不可逆的操作，而梯度信息只存在于 Tensor 对象中。
如果你需要在某个阶段将 Tensor 转为列表，并且稍后进行处理时需要保留梯度信息，那么你需要其余的策略来保存梯度信息或操作标准 Tensor 对象，而不是转化为列表。当你需要再次使用梯度信息时，你得保证操作可微分的 Tensor 而不是它们的列表/数值表示。

下面是示例代码
'''

import torch
from datasets import Dataset
import os

# 序列化张量
def tensor_to_list(tensor):
    return tensor.tolist()

# 反序列化列表为张量
def list_to_tensor(list_data, dtype=torch.float32):
    return torch.tensor(list_data, dtype=dtype)

# 构建高维张量数据集
tensor1 = torch.randn(3, 3)
tensor2 = torch.randn(3, 3)

# 使用datasets库创建数据集
# 你必须将张量转换为可序列化的格式（例如列表）
dataset = Dataset.from_dict({
    'tensor1': [tensor_to_list(tensor1)],
    'tensor2': [tensor_to_list(tensor2)]
})

# 将数据集存储为jsonl格式
jsonl_path = "tensors.jsonl"
dataset.to_json(jsonl_path)

# 从jsonl格式读取数据集
# 注意：datasets将自动推断数据字段的类型，可能需要在后处理中对其进行调整
reloaded_dataset = Dataset.from_json(jsonl_path)

# 将读取的数据集中的列还原为原始张量格式
restored_tensor1 = list_to_tensor(reloaded_dataset['tensor1'][0])
restored_tensor2 = list_to_tensor(reloaded_dataset['tensor2'][0])

# 查看恢复后的张量
print("Restored Tensors:")
print(restored_tensor1)
print(restored_tensor2)
