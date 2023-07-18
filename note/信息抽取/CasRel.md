- 不同于以往的工作中将关系看作离散的标签，我们的模型将关系建模为一种从subject到object的映射。天然具备<mark style="background: #FFB8EBA6;">解决overlap问题的能力</mark>
- **传统方法缺点：**
	- 传统的方法treat relations as discrete labels to be assigned to entity pairs. This formulation makes relation classifification a hard machine learning problem.：
	- First, the<mark style="background: #FFB8EBA6;"> class distribution is highly imbalanced.</mark>
		- 负例太多，among all pairs of extracted entities, most do not form valid relations,
	- Second, the classifier can be confused when the same entity participates in multiple valid relations (<mark style="background: #FFB8EBA6;">overlapping triples)</mark>.
		- 需要大量的样本才能训练准确
- **理论架构设计**：<mark style="background: #FFB8EBA6;">we model relations as functions that map subjects to objects</mark>
	- instead of learning relation classifiers: $f(s, o) \rightarrow r$ 
	- we learn <mark style="background: #FFB8EBA6;">relation-specific taggers</mark>:  $f_r(s)\rightarrow o$
	- <mark style="background: #FFB8EBA6;">对于给定的subject，在指定的relation下，识别出可能的object</mark>
	- 这样的框架下，triple extraction就是两步工作：先识别句子中所有可能的subjects，然后对每一个subject，使用relation-specific taggers同时识别所有可能的relation和对应的object
- **模型结构**：CASREL, an end-to-end cascade binary tagging framework
	- a BERT-based encoder module
	- a subject tagging module
	- a relation-specific object tagging module
- **效果：**
	- BERT-encoder是随机初始化时也超过前人sota
	- pre-trained BERT encoder 具有巨大提升
		- 表明了<mark style="background: #FFB8EBA6;">丰富的先验知识在三元组抽取任务的重要作用</mark>


## 3 CasRel framework
- **建模公式：**
	- ![[Pasted image 20230526143124.png]]
	- 其中 $x_j$ 是一个句子，$T_j$表示所有的三元组，D是训练集

- **模型结构细节**：
	- ![[Pasted image 20230526151241.png]]
	- BERT Encoder

	- Subject Tagger
		- it adopts <mark style="background: #FFB8EBA6;">two identical binary classifiers</mark> to detect the start and end position of subjects respectively by <mark style="background: #FFB8EBA6;">assigning each token a binary tag (0/1)</mark> that indicates whether the current token corresponds to a start or end position of a subject.
		- <mark style="background: #FF5582A6;">operation on each token:</mark>
			- ![[Pasted image 20230526144505.png]]
			- 矩阵运算的时候，两个W是可以拼在一起算的，
			- 两个p分别表示第i个token是某个subject的start位置及end位置的概率
			- 概率超过阈值置为1，否则置为0
		- <mark style="background: #FF5582A6;">解码出span：</mark>
			- 对每个预测为start的位置，寻找其后最近的end token位置，二者之间的span即为一个subject，注意在为一个start token匹配其end token的时候，不会往前找，以确保正确性。
			
	- Relation-specifific Object Taggers
		- 对<mark style="background: #FFB8EBA6;">每个subject，都要在所有relation上抽取object</mark>
		- consists of <mark style="background: #FFB8EBA6;">a set of </mark>relation-specific object taggers with the<mark style="background: #FFB8EBA6;"> same structure as subject tagger</mark>
		- 所有的object tagger同时计算
		- 唯一和subject tagger区别点是，<mark style="background: #FFB8EBA6;">object tagger的输入会加上subject feature into account </mark>
		- ![[Pasted image 20230526145932.png]]
			- 其中$v_{sub}^k$使用subject span 的vector representation的<mark style="background: #FFB8EBA6;">平均来进行融合</mark>
	- 损失loss：
		- ![[Pasted image 20230526150927.png]]

## 效果
- ![[Pasted image 20230526155028.png]]

## 总结：
- 将关系建模为一种从subject到object的映射函数
- 使用两个多个binary tagger指针网络抽取span的起止位置
- ETL-span对每个subject，一次性抽出所有object及对应的relation
- CasRel将后一步进一步分解，对每个subject，在所有的relation上做映射，得到object，模型处理的难度应该减小了，效果自然更好
- 由于对每种relation，都做一次subject抽取，结果表明，解决overlap问题能力强大