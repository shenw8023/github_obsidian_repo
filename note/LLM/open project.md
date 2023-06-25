## Chinese-LLaMA-Alpaca
- 中文LLaMA模型
	![[Pasted image 20230613144315.png]]

- 中文Alpaca模型
    ![[Pasted image 20230613144356.png]]

- LLaMA模型文件大小
	![[Pasted image 20230613145143.png]]

- 项目功能
	- 预训练（扩充词汇的中文继续预训练）
		- 1. 第一阶段：冻结transformer参数，仅训练embedding，在尽量不干扰原模型的情况下适配新增的中文词向量。
		- 2.第二阶段：使用LoRA技术，为模型添加LoRA权重（adapter），训练embedding的同时也更新LoRA参数。
	- 合并llama和lora的权重，参考示例，可以在colab中合并
		- 原版llama模型转为huggingface格式
			- 使用Transformers库的脚本：src/transformers/models/llama/convert_llama_weights_to_hf.py
			- 经过多方比对，HuggingFace模型库中的`elinas/llama-7b-hf-transformers-4.29`与原版llama模型离线转换为HF格式后的SHA256一致（已验证7B/13B/33B）
		- 合并LoRA权重，生成全量模型权重
			- 会对原版LLaMA模型（HF格式）扩充中文词表，合并LoRA权重并生成全量模型权重
	- 本地推理与部署
	- lora预训练和lora微调代码
- 问题：
	- sft训练都是<mark style="background: #FFB8EBA6;">单轮对话</mark>的形式


- 怎么扩充词汇的
	- 首先使用sentencepiece在中文语料上训练出词表
	- 然后在模型中修改原模型的input_Embedding，而transformers.PreTrainedModel 有方法实现扩充词汇：`resize_token_embeddings`
		- 具体是先根据扩充后的词表大小初始化一个Embedding，然后将旧词向量替换到新词表的前n个，剩下的那些词汇都是初始化未训练的。



torchrun #TODO 


## BELLE
- 支持lora和全量微调
- 数据集包含school_math_0.25M可以用于训练解题
- 
- 