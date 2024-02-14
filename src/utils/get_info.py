
'''
一个类get_info  ，获取语言模型推理或者训练的中间值，并保存
初始化（模型1  数据 ）#模型是huggingface的transformers加载 数据是huggingface的datasets库加载的jsonl
功能函数（实现推理过程中获取embdeing和ppl）
保存（torchsave / jsonl / faiss）

这个类首先加载tokenizer和model，然后加载数据集。
类中的 _get_embeddings_and_ppl 私有方法用于获取输入文本的embedding和PPL。
inference_and_save_info 方法遍历数据集，获取每项数据的embedding和PPL，

然后分别使用torch.save、.jsonl或者FAISS索引将它们保存到磁盘。

请确保你有一个合适的数据集文件（如data.jsonl），其中包含有 text 键的JSON对象，以及一个有效的模型名称，如 gpt2。
此外，确保output_dir存在或者你有权限在指定目录下创建文件。

注意：在实际应用中，你可能需要考虑内存和磁盘空间的限制。
对于大量数据，可能需要修改此代码以分批次处理和保存数据，以避免内存溢出。
此外，由于数据和模型可能很大，保存FAISS索引可能需要相应的硬件支持。
'''

import torch
import json
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

class ModelInfoRetriever:
    def __init__(self, model_name_or_path, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model_after_sft = self.model
        ## more...
        self.model.eval()  # Set the model to eval mode
        self.dataset = Dataset.from_json(data_path)#或者改成dataloader
        #TODO  判断传入是str 还是模型，如果传入str，按照上面进行加载，，如果传入模型。直接赋值给self#这一步方便其他地方调用

    def _get_embeddings_and_ppl(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Using the last layer hidden state

        # Calculate PPL
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
        loss = loss.view(shift_labels.size())
        ppl = torch.exp(loss.mean())

        return hidden_states, ppl.item()
    def _save_as_jsonl(self, output_dir):
        # 序列化 from .tensor2jsonl import xx
        
        # 存jsonl
        pass
      
    def inference_and_save_info_pipeline(self, output_dir):
        embeddings_collection = []
        ppls = []

        for entry in self.dataset:
            input_text = entry['text']
            embedding, ppl = self._get_embeddings_and_ppl(input_text)
            embeddings_collection.append(embedding.cpu().numpy())
            ppls.append(ppl)

      

        # Save embeddings and PPL locally
        torch.save(torch.stack(embeddings_collection), f'{output_dir}/embeddings.pt')
        with open(f'{output_dir}/ppls.jsonl', 'w') as f:
            for item in ppls:
                f.write(json.dumps(item) + '\n')

        ## If you want to use FAISS for saving embeddings:
        #embeddings_matrix = torch.stack(embeddings_collection).view(-1, embedding.size(-1)).numpy()
        #index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
        #index.add(embeddings_matrix)
        #faiss.write_index(index, f'{output_dir}/faiss_index.bin')

# Usage
model_name = 'gpt2'  # specify the model name
data_path = 'data.jsonl'  # specify the path to your JSONL file
output_dir = 'model_info_output'  # specify the output directory

info_retriever = ModelInfoRetriever(model_name, data_path)
info_retriever.inference_and_save_info_pipeline(output_dir)
