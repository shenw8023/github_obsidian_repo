## Introduction 

- Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy

- 传统的pipeline方式extract-then-classify，忽略了两个子任务之间的关联信息，分类器会受限于大量的负例entity_pairs。
- 具体方式总结：
		- we first decompose the joint extraction task into two interrelated subtasks, namely<mark style="background: #FF5582A6;"> HE extraction</mark> and <mark style="background: #FF5582A6;">TER extraction</mark>.The former subtask is to distinguish all head-entities that may be involved with target relations, and the latter is to identify corresponding tail-entities and relations simultaneously for each extracted head-entity.
- 理论依据：
	- <mark style="background: #FF5582A6;">extract-then-label(ETL)</mark> paradigm can be understood by decomposing the joint probability of triplet extraction into conditional probability：
	- $$p(h,r,t|S) = p(h|S)p(r,t|h,S)$$
	- where $(h,r,t)$ is a triplet in sentence S
- 优点：
	- 这样的范式能让TER阶段利用已经获取的head-entity的<mark style="background: #FFB8EBA6;">语义信息和位置信息</mark>
	- 同时一个head-entity能和多种tail-entity交互，进而解决<mark style="background: #FFB8EBA6;">overlapping relations</mark>
	- 跟extract-then-classify方式比较，本文方法第一步的时候不需要一次性抽出所有的entity，<mark style="background: #FFB8EBA6;">只有跟目标三元组相关的head-entity才会被抽取</mark>，缓解大量无关entity-pairs的问题。
- 受抽取式阅读理解的启发，进一步将HE和TER extraction分解为 a span-based tagging scheme
	- a hierarchical boundary tagger
	- a multi-span decoding algorithm.


## 2 Methodology
- 2.1 Tagging Scheme
	- HE extraction
		- 第一个序列标注来identify the start position of one head-entity
		- 第二个序列标注来identify the end position of one head-entity
		- 标签范围是 {entity_type , O}，O表示非实体
		- 该步骤既确定了head-entity的位置信息，也同时给出了实体类型
	- TER extraction
		- 针对每一个识别的head-entity
		- 第一个序列标注确定该head-entity对应的tail-entity的start位置
			- 标签范围是 {relation_type, O}
		- 第二个序列标注确定该tail-entity的end 位置（感觉应该也会同时标出relation_type，类似HE步骤，start和end位置的relation_type可以做一致性验证）
	- 图示：![[Pasted image 20230525204025.png]]
- 2.2 Hierarchical Boundary Tagger
	- 上述HE和TER步骤本质上是类似的，可以抽象为一种通用的extractor module，称为<mark style="background: #FFB8EBA6;">hierarchical boundary tagger (HBT)</mark>，不管是head-entity还是抽tail-entity都可以视为targets，那么可以定义：
		- 从句子S中抽取出<mark style="background: #FF5582A6;">标签为 l 的目标 t </mark>的概率为：
		- $$p(t,l|S) = p(s_t^l|S) p(e_t^l|s_t^l,S)$$
		- 其中$s_t^l$表示start index of $t$ with label $l$, $e_t^l$表示end index
	- 这样的分解方式表明子任务之间是有<mark style="background: #FFB8EBA6;">天然的顺序信息</mark>的：预测end位置可能从已经预测出的start位置受益。网络设计时，也要尽可能的将<mark style="background: #FF5582A6;">较低层的输出同时给到较高层的输入</mark>。





## 网络结构总结
- 整体来看，做了两次分解任务
	- <mark style="background: #FF5582A6;">HE + TER：</mark>先抽head-entity，然后利用head-entity信息同时抽出tail-entity 以及 relation-type
		- $p(h,r,t|S) = p(h|S)p(r,t|h,S)$
		- ![[Pasted image 20230525214757.png]]
		- 底层一个BiLSTM做share encoder，编码全局语义信息
		- 图中$g$是全局视角的向量信息，对应下方公式的$a_i$
		- 
		
	- <mark style="background: #FF5582A6;">HBT：</mark>进一步分解为先抽目标的start位置以及label，再抽end位置以及label
		- $p(t,l|S) = p(s_t^l|S) p(e_t^l|s_t^l,S)$
		- ![[Pasted image 20230525213816.png]]
		- 注意$a_i$是一个用于传递全局信息的量，它在HE阶段是输入序列向量的max pooling结果，然后每个位置进行复制；它在TER阶段是global representation向量和head-entity 进行fuse后的向量 二者拼接
		- $p_i^{se}$ 是一个可训练的位置矩阵，根据当前position及往前的最近的start-position的相对位置来确定：C设为最大序列长度
			- ![[Pasted image 20230525214558.png]]
		- <mark style="background: #FFB8EBA6;">start序列标注过程：</mark>
			- ![[Pasted image 20230525214853.png]]
		- <mark style="background: #FFB8EBA6;">end序列标注过程：</mark>
			- ![[Pasted image 20230525214910.png]]
	- <mark style="background: #FF5582A6;">LOSS：</mark>
		- ![[Pasted image 20230525215235.png]]
		- 本质就是<mark style="background: #FFB8EBA6;">序列标注的交叉熵</mark>，对每个token，它的真实tag分布为[1,0,0,0]，他的预测分布为[0.9,0.1,0.2,0.1]，交叉熵就是$1*log(0.9)$，也就是只看这个token的真实tag的预测概率的对数；这里的sta和end上标分别表示在start序列中和end序列中计算。