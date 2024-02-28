# 数据数量
Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases (Arxiv, Mar. 2023) [Paper]  
Lima: Less is more for alignment (Arxiv, May 2023) [Paper]  
Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning (Arxiv, May 2023) [Paper]  
Scaling Relationship on Learning Mathematical Reasoning with Large Language Models (Arxiv, Aug. 2023) [Paper] [Code]  
How Abilities In Large Language Models Are Affected By Supervised Fine-Tuning Data Composition (Arxiv, Oct. 2023) [Paper]  
Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace (Arxiv, Oct. 2023) [Paper]  

# 数据质量
## 指令质量

Self-refine: Iterative refinement with self-feedback (Arxiv, Mar. 2023) [Paper][Project]  
Lima: Less is more for alignment (Arxiv, May 2023) [Paper]  
Enhancing Chat Language Models by Scaling High-quality Instructional Conversations (Arxiv, May 2023) [Paper] [Code]  
SelFee: Iterative Self-Revising LLM Empowered by Self-Feedback Generation (Blog post, May 2023) [Project]  
INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models (Arxiv, Jun. 2023) [Paper] [Code]  
Instruction mining: High-quality instruction data selection for large language models (Arxiv, Jul. 2023) [Paper]  
Harnessing the Power of David against Goliath: Exploring Instruction Data Generation without Using Closed-Source Models (Arxiv, Aug. 2023) [Paper]  
Self-Alignment with Instruction Backtranslation (Arxiv. Aug. 2023) [Paper]  
SELF: Language-Driven Self-Evolution for Large Language Models (Arxiv, Oct. 2023) [Paper]  
Automatic Instruction Optimization for Open-source LLM Instruction Tuning (Arxiv, Nov. 2023) [Paper] [Code]  
## 指令多样性

Self-instruct: Aligning language models with self-generated instructions (ACL 2023) [Paper][Code]  
Stanford Alpaca (Mar. 2023) [Code]  
Enhancing Chat Language Models by Scaling High-quality Instructional Conversation (Arxiv, May 2023) [Paper] [Code]  
Lima: Less is more for alignment (Arxiv, May 2023) [Paper]  
InsTag: Instruction Tagging for Analyzing Supervised Fine-Tuning of Large Language Models (Arxiv, Aug. 2023) [Paper] [Code]  
Explore-Instruct: Enhancing Domain-Specific Instruction Coverage through Active Exploration (Arxiv, Oct. 2023) [Paper] [Code]  
DiffTune: A Diffusion-Based Approach to Diverse Instruction-Tuning Data Generation (NeurIPS 2023) [Paper]  
Data Diversity Matters for Robust Instruction Tuning (Arxiv, Nov. 2023) [Paper]  
## 指令复杂度/难度
WizardLM: Empowering Large Language Models to Follow Complex Instructions (Arxiv, April 2023) [Paper] [Code]  
WizardCoder: Empowering Code Large Language Models with Evol-Instruct (Arxiv, Jun. 2023) [Paper] [Code]  
Orca: Progressive Learning from Complex Explanation Traces of GPT-4 (Arxiv, Jun. 2023) [Paper] [Code]  
A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment (Arxiv, Aug. 2023) [Paper]  
Can Large Language Models Understand Real-World Complex Instructions? (Arxiv, Sep. 2023) [Paper] [Benchmark]  
Followbench: A multi-level fine-grained constraints following benchmark for large language models (Arxiv, Oct. 2023) [Paper] [Code]  




# paper_to_src
2402.10430  Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models  
「LP指标」具体来说，LP是一个衡量样本在模型训练过程中学习难度的指标。对于每个样本，LP计算的是该样本在第一个训练epoch结束时困惑度（perplexity）的下降量与整个训练过程中困惑度总下降量的比值。  
如果一个样本在训练开始时的困惑度是P0，在第一个epoch结束时的困惑度是P1，那么LP(1)的计算公式为：LP(1) = (P1 - P0) / (P0 - Pn)，其中Pn是训练结束时的困惑度。这一方法允许模型根据早期学习情况自主评估样本难度，并选择那些在训练初期学习较少（即困难）的样本进行训练。

2402 LESS: Selecting Infuential Data for Targeted Instruction Tuning  
LESS（Low-rank gradiEnt Similarity Search） 利用梯度信息估算单个训练数据点影响的启发，研究人员设计了一种优化器感知方法来选择这些数据。并使用lora优化

2308 From Quantity to Quality Boosting LLM Performance with Self-Guided Data Selection  
测量并比较模型对给定指令生成回复的能力和模型直接生成回复的能力，通过计算"指令跟踪难度"（IFD）得分，作为数据筛选的条件，得分越高，说明质量更高。

2401  An Experimental Design Framework for Label-Efficient Supervised Finetuning of Large Language Models  
主动学习-选择最具信息量的样本进行学习








