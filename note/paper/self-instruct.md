## 主要内容
- SELF-INSTRUCT：
	- a semi-automated process for instruction-tuning a pretrained LM using instructional signals from the model itself.
- 
	- 175条人写的种子instruction
	- 迭代生成52k instructions, 以及 82k 问答对
	- 每轮迭代都经过多种方式做<mark style="background: #FFB8EBA6;">清洗和去重</mark>才能入池
	- 50% of them have less than 0.3 ROUGE-L overlaps with the seed instructions
- 使用的模型：GPT-3 （GPT-3生成能力有这么强吗？） #思考
- 比较对象：text-davinci-001 （只做过supervised fine-tuning with human demonstrations.）
- 比较结果：
	- SUPERNI 结果显示超过 GPT3  (+33.1%)
	- outperforming models trained on other publicly available instruction datasets 
	- leaving only a 5% gap behind InstructGPT001.
- 指令数据的多样性和数量，直接影响模型对新任务的generalizability
- 文章起源：Our work aims to tackle this bottleneck by reducing the dependence on human annotators.

## relate work
- Language models for data generation and augmentation.
	- 跟前人的区别：
		- our work differs from this line in that it is not specifific to a particular task. A distinct motivation for SELF-INSTRUCT is to bootstrap new task definitions that may not have been defined before by any NLP practitioner (though potentially still important for downstream users).

- 具体步骤
	1. Instruction Generation
		- 基于一个结论：大型预训练模型的in-context learning能力
		- 初始task pool包含175条人写的指令（1 instruction and 1 instance for each task）
		- 每一步，从pool中随机采样8个instructions作为in-context（其中6条是人写的，2条是之前step中模型生成的指令），对模型提问，要求模型生成多条新的instruction。
			- ![[Pasted image 20230419205533.png]]
	2. Classifification Task Identifification
		- 对GPT3生成的新的指令区分是分类型还是非分类型，有两种处理方式，所以首先对指令做区分
		- 使用池子中12个分类型指令和19个非分类型指令构建in-context，直接让GPT-3做few-shot判断
			- ![[Pasted image 20230419210226.png]]
			- ![[Pasted image 20230419210239.png]]
- 
- 
	3. Instance Generation
		- 根据instructions和其task type，为每个instruction生成instances
		- 由于PLM具备根据instruction-input-output 形式的in-context examples生成文本的能力，所以采用PLM生成instances
		- 对分类型的instruction和非分类型的采取不同的生成方式
			- 非分类型任务采用<mark style="background: #FFB8EBA6;"> Input-first Approach</mark>
				- 也就是采用in-context learning方式直接让GPT3根据指令，生成多个instances
				- ![[Pasted image 20230420193330.png]]
				- ![[Pasted image 20230420193351.png]]
			- 分类形式的任务采用<mark style="background: #FFB8EBA6;">Output-first Approach</mark>
				- 具体 先生成指令可能的所有类别标签，然后让模型为每个标签生成一个问题（first generate the possible class labels, and then condition the input generation on each class label.）
				![[Pasted image 20230420194232.png]]
				![[Pasted image 20230420194256.png]]
- 
	4. Filtering and Postprocessing.
		- 新入池的instruction 
			- when its ROUGE-L overlap with any existing instruction is less than 0.7.
			- exclude instructions that contain some specifific keywords (e.g., images, pictures, graphs)
			- filter out instances that are exactly the same or those with the same input but different outputs.
- ![[Pasted image 20230420204824.png]]


## 结果
- Zero-Shot Generalization on SUPERNI benchmark
	![[Pasted image 20230420213502.png]]
- Human evaluation setup
	![[Pasted image 20230420213754.png]]

## Limitations
- Tail phenomena
	- self-instruct 依赖一个大模型  LMs’ largest gains correspond to the frequent uses of languages (head of the language use distribution), and there are minimal gains in the low-frequency contexts.
	- As a consequence, the approach might show brittleness with respect to uncommon and creative instructions.
- Reinforcing LM biases


## 细节记录
- every finetuning experiment is trained for <mark style="background: #FF5582A6;">two epochs </mark>to avoid overfitting the training tasks
- <mark style="background: #FF5582A6;">reducing the number of instances per task </mark>does not degrade the model’s generalization performance to unseen tasks.
-  we conjecture that future work can largely benefit from<mark style="background: #FF5582A6;"> improving the quality of our generated data</mark> by using human annotators or training a reward model to select better generations