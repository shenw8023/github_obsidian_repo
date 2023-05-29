
## 分类
- Pipeline方法：
	- 拆分为两个子任务：实体识别和关系分类
		- 先抽取所有实体，再对实体对进行关系分类，两个任务依次执行，之间没有交互。
		
- 联合模型：
	- 两个子任务之间实现信息交互，提升效果
	- 基于参数共享：
		- 解码时spo三元组不是一起出来的，过程和pipeline类似，分步完成；区别是参数共享loss是各个子过程的loss之和，反向传播更新参数的时候会更新所有子过程的参数，而pipeline方法各子任务之间没有联系
	- 基于联合解码：
		- 解码时spo三元组是一起出来的，更符合联合的思想
		
- 解码方式对性能影响也很大：
	- 序列标注、指针网络、片段分类
	
- ![[Pasted image 20230525104737.png]]	




## 参数共享和联合解码方法
- ![[Pasted image 20230525105115.png]]


- 经典模型
	- 参数共享：
		- **End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures** #todo
			- 两个LSTM模型，一个实体识别，一个关系分类，应用依存树

	- 联合解码：
		- **Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme**
			- 论文的主要工作是设计了一种特别的标签类型，将实体识别和关系分类转化为序列标注问题。
			- 将实体位置信息，关系类型，实体角色信息，三个标签拼出一个标签类型，做<mark style="background: #FF5582A6;">序列标注</mark>，直接得到结果进行解码
			- 编码层使用Bi-LSTM来进行编码；之后，解码层再使用 LSTMd( 作者提出的一种基于LSTM的改进型 ) 进行解码
			- <mark style="background: #FF5582A6;">问题：</mark>序列标注的通病，不能解决实体或关系的重叠问题。
			- ![[Pasted image 20230525105802.png]]
			- ![[Pasted image 20230525105907.png]]
	
- 优缺点对比：
	- <mark style="background: #FFB8EBA6;">pipeline方式</mark>：
		- 缺点：错误传播，忽略了子任务之间的联系
	- 参数共享
		- 优点：1.实体识别和关系分类过程不是同步的，所以后面步骤的训练可以使用前面步骤的结果作为特征，2.<mark style="background: #FF5582A6;">灵活性更高</mark>，能够更好的处理重叠实体关系<mark style="background: #FF5582A6;">重叠问题</mark>。	
		- 缺点：实体识别和关系分类仍然是两个独立的过程，没有真正的“联合”。
	
	- 联合解码
		- 优点：没有明确的将抽取过程分为多个步骤，真正实现了实体识别和关系分类的信息共享。
		- 缺点：目前要实现联合解码<mark style="background: #FF5582A6;">一般需要设计复杂的标签或者解码过程</mark>，而且联合解码的方式对重叠的关系三元组的识别效果不是很好。



- 解码方式对比：
	- 序列标注：
		- 主要缺点：不能解决<mark style="background: #FF5582A6;">嵌套问题</mark>
		- 序列标注方法在命名实体识别任务中非常常用，但是在实体关系抽取任务中，序列标注有非常多的不足，很重要的一点是一个token只能有一个标签，而关系抽取任务情况复杂，存在实体重叠、关系重叠等诸多特殊情况，经常需要设计比较复杂的联合标签才能完成对实体关系三元组的抽取，然而标签越复杂，抽取效果就越差。因此最新的论文基本都不再采用这种方法。
		
	- 指针网络：
		- 针对一种实体类型或关系类型，分别使用头指针，尾指针预测两个标签序列
		- <mark style="background: #FF5582A6;">使用多个标签序列（多层label网络）来表示一个句子</mark>。例如采用两个标签序列，一个表示实体的起始位置，另一个表示实体的结束位置。在解码时使用Sigmoid代替Softmax，预测每个token对应的标签是0还是1。
		- 由于指针网络的<mark style="background: #FF5582A6;">表达能力较强</mark>，可以很好地<mark style="background: #FF5582A6;">解决实体-关系重叠等问题</mark>，所以在目前的实体关系抽取方法中被大量使用。虽然指针网络是一个比较好的方案，但是在指针网络的使用过程中很可能会遇到<mark style="background: #FF5582A6;">标签不平衡问题</mark>，需要进行一些调参工作。
		
	- 片段排列：
		- 找出所有可能的片段组合，然后针对每一个片段组合求其是否是实体的概率，在确定所有的实体之后，对所有实体两两配对，然后求每一对实体对之间存在关系的概率。
		- 如果有N个实体，M种关系，那么存在N × N个实体对（实体对是有序的），需要求N × N × M个概率，来判断每个实体对之间的关系
		- 需要<mark style="background: #FF5582A6;">限制span长度并合理削减负样本</mark>
	
	- 优缺点对比：
		- 序列标注：简单，效果好，不能解决<mark style="background: #FF5582A6;">嵌套问题</mark>
		- 指针网络：能解决嵌套问题，存在标签不平衡问题，且<mark style="background: #FF5582A6;">解码层后还要对多个标签序列做解析确定实体</mark>
		- 片段排列：不需要从标签到实体的转换过程，但是大量负标签带来的<mark style="background: #FF5582A6;">计算量问题</mark>



## 重叠问题
- 没有重叠的Normal类型，关系重叠的entity-pair-overlap（subject和object同时被两个关系使用）类型，实体重叠的single-entity-overlap（单个实体被重复使用）类型。
	- ![[Pasted image 20230525110352.png]]
	



## 评估方式
- 一般一个完全准确的三元组才视为一个正确预测
- 对标准三元组和预测三元组，计算precision, recall, f1
- 参考bert4torch的代码
- 


## Benchmark
- WebNLG https://paperswithcode.com/sota/relation-extraction-on-webnlg
- 



## 主要SOTA模型

- [[ETL-Span]]
- [[CasRel]]
- 