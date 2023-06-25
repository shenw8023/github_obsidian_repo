- <mark style="background: #FFB8EBA6;">论文：GPT understands too</mark>
- <mark style="background: #FFB8EBA6;">可同时应用于NLG和NLU任务</mark>
- we show that GPTs can be better than or comparable to<mark style="background: #FFB8EBA6;"> similar-sized BERTs on NLU tasks</mark> with a novel method P-tuning
- superglue benchmarks #TODO 
- P-tuning also improves BERTs’ performance in both few-shot and supervised settings while largely reducing the need for prompt engineering.
- P-tuning– to automatically search prompts in the continuous space to bridge the gap between GPTs and NLU applications.
- It also suggests that language models contain much more world knowledge and prior task knowledge than we previously assumed.
 - <mark style="background: #FF5582A6;">P-Tuning 只在 embedding 层增加参数，而 Prefix-Tuning 在每一层都添加可训练参数 。</mark>
## motivation
- 大模型通常效果更好，但是其迁移能力不行
- 人工构建离散prompt虽然能一定程度上适配下游任务，但是严重依赖于验证集，表现也不稳定，prompt中一个字的变化可能带来任务效果巨大的变化
- we delve into the problem of finding continuous prompts that can be differentially optimized.


- 直接优化连续的 prompt 参数面临两个挑战：
	- 一是预训练模型原始的词向量已经高度离散，若随机初始化 prompt 向量并进行 SGD 优化，也只会在小范围内优化并陷入局部最小值；
	- 二是 prompt 向量之间是相互关联而不是独立的。
	- 论文中 **设计了一个 prompt 编码器，该编码器由一个 Bi-LSTM 和一个两层的前馈神经网络组成，对 prompt embedding 序列进行编码后再传入到语言模型中** 。

- 本质思考
	- 固定预训练模型参数，引入少量额外参数，用于下游任务的迁移学习
	- 例如 prefix_tuning, p_tuning, lora 都是类似的
- 参考苏剑林：<mark style="background: #FFB8EBA6;">https://kexue.fm/archives/8295</mark>
	- 为什么有效
	- lm+全连接为什么能解决下游任务
	- 一点原实验细节：https://github.com/THUDM/P-tuning/issues/5
	- 按照作者的意思，LSTM是为了帮助模版的几个token（某种程度上）更贴近自然语言，但这并不一定要用LSTM生成，而且就算用LSTM生成也不一定达到这一点。笔者认为，更自然的方法是在训练下游任务的时候，不仅仅预测下游任务的目标token（前面例子中的“很”、“新闻”），还应该同时做其他token的预测。
	- 比如，如果是MLM模型，那么也随机mask掉其他的一些token来预测；如果是LM模型，则预测完整的序列，而不单单是目标词。这样做的理由是：因为我们的MLM/LM都是经过自然语言预训练的，所以我们（迷之自信地）认为能够很好完成重构的序列必然也是接近于自然语言的，因此这样增加训练目标，也能起到让模型更贴近自然语言的效果。经过笔者的测试，加上这样辅助目标，相比单纯优化下游任务的目标，确实提升了效果。

- **P-Tuning V2** 方法的思路其实和 Prefix-Tuning 相似，在 **模型的每一层都应用连续的 prompts** 并对 prompts 参数进行更新优化。同时，该方法是 **针对 NLU 任务优化和适配** 的。