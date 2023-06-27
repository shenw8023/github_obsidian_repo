![[Pasted image 20230626204443.png]]

- 论文标题：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2101.00190.pdf)
- <mark style="background: #FF5582A6;">仅用于 NLG 任务</mark>
- Prefix-tuning prepends a sequence of continuous task-specific vectors to the input, which we call a prefix
- Extending this intuition beyond generating a single word or sentence, we want to find a context that steers the LM to solve an NLG task.
- Intuitively, the context can influence the encoding of x by guiding what to extract from x; and can influence the generation of y by steering the next token distribution.
- For subsequent tokens, the Transformer can attend to the prefix as if it were a sequence of “virtual tokens”, but unlike prompting, the prefix consists entirely of free parameters which do not correspond to real tokens.
- prefix-tuning <mark style="background: #FFB8EBA6;">only optimizes the prefix</mark>
- <mark style="background: #FFB8EBA6;">Controllable generation</mark> aims to steer a pretrained language model to match a sentence level attribute.
- 在模型输入前添加一个连续的向量序列
- prefix tuning 在 transformer 每一层都有可训练参数。
- 效果：
	- 在全量数据上和达到微调的效果
	- 在少样本数据时，比微调更好
	- 零样本推理能力更强

- <mark style="background: #FF5582A6;">具体方式：</mark>
	- 构建一个prefix矩阵用于训练，训练过程固定LLM参数，只优化prefix矩阵：输入为 [prefix；X]，输出为 Y
	- 前向计算的时候，prefix部分的序列直接从复制过来（不是通过模型参数attention计算得到）；X 部分的序列通过模型参数计算得到（计算的过程中每个token都会attention到prefix序列，所以prefix会影响后续所有token的表示，也是prefix比infix好的原因）
		- ![[Pasted image 20230626203526.png]]
	- <mark style="background: #FF5582A6;">不仅输入的embedding层添加prefix，而且模型的每一层计算都要添加prefix序列，且都是各不相同的，都要在训练阶段做优化。</mark>如果只在embedding层加prefix，后续网络每层的激活都是正常计算，那就是论文中说的<mark style="background: #FFB8EBA6;">embedding-only方式</mark>，效果会略差。
	- 为了训练稳定，通过对prefix矩阵对应的参数会通过一个全连接网络来处理一下，实际训练完成后，直接丢弃这个全连接网络，保留最终的prefix矩阵就行了
	- 


- AutoPrompt 
	- (Shin et al., 2020) searches for a sequence of discrete trigger words and concatenates it with each input to elicit sentiment or factual knowledge from a masked LM.
- our method optimizes <mark style="background: #FFB8EBA6;">continuous prefixes</mark>, which are more expressive (§7.2); moreover, we focus on <mark style="background: #FFB8EBA6;">language generation tasks</mark>.
- 示例了两种形式的模型怎么使用prefix-tuning
	- autoregressive decoder only
	- encoder-decoder
- Based on intuition from prompting, we **believe that having a proper context can steer the LM without changing its parameters.**
- the language model parameters φ are fixed and the prefix parameters θ are the only trainable parameters.
- 直接优化prefix会不稳定，very sensitive to the learning rate and initialization.使用MLP来重参数化
- 结论：
	- 0.1%的参数训练，可以达到甚至超过fine_tune的效果
	- 更少的参数量，更好的效果
	- we observe that prefix tuning has a comparative advantage when the number of training examples is smaller
	- outperforms fine-tuning in low-data regimes. In addition to requiring many fewer parameters, but the gap narrows as the dataset size increases.


- prefix length的选择：
	- ![[Pasted image 20230516204124.png]]
	- 左边是summerization，右边是table-to-text
- full vs embedding-only
	-  Embedding-only 方法只在 embedding 层添加前缀向量并优化，<mark style="background: #FFB8EBA6;">而 Full 代表的 Prefix-tuning 不仅优化 embedding 层添加前缀参数，还在模型所有层的激活添加前缀并优化。</mark>
	- **表达能力增强链条：discrete prompting ＜ embedding-only ＜ prefix-tuning**
- prefix的初始化很重要
- prefixing 效果好于 Infixing
- 