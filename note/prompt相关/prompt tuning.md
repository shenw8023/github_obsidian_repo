![[Pasted image 20230626205529.png]]

- 论文：[The Power of Scale for Parameter-Efficient Prompt Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.08691.pdf)
- In this paper, we propose prompt tuning as a further simplification for adapting language models. 
	- We freeze the entire pre-trained model and only allow an additional k tunable tokens per downstream task to be prepended to the input text. This “soft prompt” is trained end-to-end and can condense the signal from a full labeled dataset, allowing our method to <mark style="background: #FFB8EBA6;">outperform few-shot prompts and close the quality gap with model tuning</mark>
- 还有一个工程上的优势：
	- 全量微调要求每个下游任务部署一个大模型，而论文的方式只需要将不用的prompt拼接到不同的任务输入前面，构建一个mixed_task_batch，然后送到一个LLM做计算，可以在一次batch推理中做不同下游任务的计算
- 模型参数越大，fine-tuning和promt-tuning在下游任务的差距越小，10B的参数量，两者持平
- <mark style="background: #FFB8EBA6;">显式将任务特定参数从模型通用能力参数中分离出来</mark>，有很多额外的好处，对领域迁移有更好的弹性
- <mark style="background: #FF5582A6;">learning multiple prompts for the same task</mark>, can boost quality and is more effificient than classic model ensembling.

- 对比实验
	- prompt长度实验：越长的prompt能带来越好的效果，但模型规模足够大，长度影响就很小 【可能是因为越长的prompt对应越长的参数表达空间，效果自然会变好，但模型规模足够大，则这部分的参数占比可能就不明显】
	- prompt初始化策略：随机初始化 < 最长词汇初始化 < 标签字符串初始化，但是模型规模变大后，则初始化策略也无关重要【整体上就是大力出奇迹，几乎可以忽略所有的超参数细节】
	- Domain-shit跨域实验，prompt方法会比微调模型效果更优泛化效果，尤其域间数据差异越大，效果越明显【这主要反映的是prompt方式在下游任务使用中通用性更强，调整下游任务时候，损失的语言模型本身能力较小】
- 点评：
	- 这篇论文主要是针对T5模型为backbone，提出prompt微调方式训练下游任务 ；与prefix-tuning相比，主要是实验的模型架构不同，并设计了多重对比实验，探索了不同初始化方式，不同prompt对语言模型微调的影响
	- 最终整体的结论：随着模型规模变大，prompt的初始化和长度对效果影响很小，鲁棒性和泛化能力也有比较好的提升 整体论文没啥创新点，主要实验多