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
	- 例如 prex_tuning, p_tuning, lora 都是类似的
- 参考苏剑林：<mark style="background: #FFB8EBA6;">https://kexue.fm/archives/8295</mark>
	- 为什么有效
	- lm+全连接为什么能解决下游任务
	- 一点原实验细节：https://github.com/THUDM/P-tuning/issues/5

- **P-Tuning V2** 方法的思路其实和 Prefix-Tuning 相似，在 **模型的每一层都应用连续的 prompts** 并对 prompts 参数进行更新优化。同时，该方法是 **针对 NLU 任务优化和适配** 的。