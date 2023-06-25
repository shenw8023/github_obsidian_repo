## Abstract
- 数据量越来越大，把控数据质量越来越难
- We leverage a largely ignored source of information: the behavior of the model on individual instances during training (training dynamics) for building data maps.
- 对每个样本使用两个符合直觉的度量：
	- the model’s confidence in the true class, 
	- and the variability of this confidence across epochs
- 在四个数据集上展示了该模型无关的衡量方式揭示了数据分布中的三个不同的区域
	- 关于模型的 ambiguous regions，该区域对模型的分布外泛化能力贡献最多
	- 最大范围的是 easy to learn regions，在模型优化过程中扮演重要角色
	- hard to learn regions，这部分通常跟错误标注有关
- 文章结论证明，更加关注数据质量而不是数量能带来更鲁棒的模型和提高分布外泛化的能力 out-of distribution generalization

## Introduction
- 业界共识：标注数据越多，模型学习多样化的能力越强，也就带来模型更好的泛化能力
- 实际应用中，out-of-distribution (OOD) generalization还是有很大的挑战，尽管大型预训练模型能缓解，但不能完全解决，这需要我们更仔细研究数据集，因为并不是所有的样本对模型的学习贡献都是相等的。
- 然而数据集越大，越难以评估数据质量，How can we automatically characterize data instances with respect to their role in achieving good performance in- and out-of-distribution?
- 从绘图学中得到启发
	- 为实例找到它在整个数据集中的坐标，提出了**data maps** :  一个基于模型的，用于关联数据集中样本的工具。
	- 该data maps的坐标构建使用**training dynamics** : 模型在训练过程中的行为表现
	- 考虑使用对gold label预测概率的均值和标准差，即：confidence and variability（置信度和可变形）
	- 结合图1说明:
		- ![[Pasted image 20230620202936.png]]
		- ambiguous区域：对样本的true class 预测概率波动很大<mark style="background: #FFB8EBA6;"> (high variability)</mark>，ambiguous for the model
		- easy-to-learn区域：预测总是正确且一致<mark style="background: #FFB8EBA6;"> (high confifidence, low variability)</mark>
		- hard-to-learn区域：大部分数据后来发现都是错误标注的<mark style="background: #FFB8EBA6;"> (low confidence, low variability)</mark>
	- Our findings indicate that data maps could serve as effective tools to diagnose large datasets, at the reasonable cost of training a model on them. Locating different regions within the data might pave the way for constructing higher quality datasets., and ultimately models that generalize better.

## 2 Mapping Datasets with Training Dynamics
- 目标是构建数据集的Data Maps，帮助可视化数据集，发现不同类型的样本对模型学习的贡献是怎么样的
### 2.1 Training Dynamics
- 第i个样本的training dynamics是经过全部的训练epochs得到
- 该数值然后作为map中的坐标
- 第一个度量是**confidence**
	- 度量模型有多大的置信度赋予true label
	- We define **confidence** as the mean model probability of the true label $y_i^*$ across epochs:
	- ![[Pasted image 20230620204205.png]]
	- <mark style="background: #FFB8EBA6;">其中p是模型在每个epoch后的预测概率，并且注意μ是关于真实标签的，而不是关于模型预测的最高分数对应label</mark>
- 定义 **correctness**指标：
	- 在所有epoch中，样本被模型正确预测gold label的次数的比例
- 另一个度量定义为 **variability**：
	- 用于度量在所有epochs上对真实标签预测概率的分散程度，使用标准差
	- ![[Pasted image 20230620205510.png]]
	- 注意，also depends on the gold label
	- 对一个样本来说，模型总是预测相同的标签（不管对错）将获得高variability，相反模型预测摇摆不定的样本，会获得低variability
- 补充说明：
	- confidence和variability都是基于epochs计算的，但是都可以基于更细的粒度，例如. over every few steps of optimization. 能提高效率。
	- 但是要注意忽略一开始的几个steps，要直到模型优化稳定的时候再开始统计，如果是基于epoch的话，第一个epoch可以使用是因为已经经过很多步的优化了。

### 2.2 Data Maps
- 上面图1
	- 占大部分区域的样本是属于 high confifidence and low variability (top-left)；模型对这些样本总是以高置信度正确预测，因此把他们称为 easy-to-learn
	- 第二小一点的区域是 with low variability and low confidence (bottom-left cornor)，这些区域样本很少能正确预测，因此称为 hard-to-learn
	- 第三个区域包含ambiguous 样本，with high variability (right-hand side)，模型对这些样本的预测摇摆犹豫的，因此称为 ambiguous
- 图2
	- ![[Pasted image 20230620212506.png]]
	- 值得注意的是，和图1最大的区别是 hard-to-learn 区域要小得多，可能的解释是该数据集被严格审核过
	- <mark style="background: #FFB8EBA6;">不同的模型结构可能会带来不同的 data maps，尽管类似的模型，得到的data maps看起来类型，但是一个实例所属的region会有所不同。</mark>


## 3 Data Selection using Data Maps
- Data maps揭示了数据集的不同区域，很自然我们会思考不同区域的样本在学习和泛化的过程中扮演了什么样的角色？
- 为了回答这个问题，我们分别使用不同区域的样本来训练不同的模型，使用standard in-distribution (ID), as well as out-of-distribution (OOD) evaluation
- 数据子集构建：
	- 两个baseline
		- 全量数据
		- 33%随机采样数据
	- high-confidence (ambiguous)
		- 根据confidence指标，从高往低取前33%的样本
		- 对应easy-to-learn区域
	- low-confidence (hard-to-learn)
	- low-variability 
		- 根据variability指标，从高往低取前33%的样本
		- 对应comprises a mixture of easy-to-learn and hard-to-learn examples.
	- high-correctness
	- low-correctness
- 测试集构建
	- ID test sets是从原始数据集中
	- OOD test sets are either manually or semi-automatically created.
	- 所有的评测指标为accuracy
	- 
- 结果
	- 最优的结果是ambiguous和hard-to-learn区域，甚至超过全量数据的 baseline（尽管只用了1/3的数据）
	- high-confidence和high-correctness数据的结果都比较差
	- highlight the important role played by examples which are challenging for the model, i.e., ambiguous and hard-to-learn examples

## 4 Role of Easy-to-Learn Instances
- . This indicates that ambiguous instances alone might be insufficient for learning.
- . This indicates that for successful optimization, it is important to include easier-to-learn instances.
- Selection of the optimal balance of easy-to-learn and ambiguous examples in low data regimes is an open problem;

## 5 Detecting Mislabeled Examples

- 我们猜想hard-to-learn(low confidence)区域的数据很可能是存在错标
- 验证实验：
	- 混入一下错标数据重新训练模型，观察这些混入数据的data maps坐标变化
	- 混入数据是从easy-to-learn采样的数据，然后取反进行标错，混入到训练集；没有从hard-to-learn采样是避免一些我们不知道的错标数据，影响观察
	- 实验结论：Flipped instances move to the lower confidence regions after retraining, with some movement towards higher variability. This indicates that perhaps the hard-to-learn region (low confidence) of the map contains other mislabeled instances.

- 证明了猜想，下一步就是去hard-to-learn区域寻找所有可能的错标数据
	- 这里采取训练一个简单的线性分类器来自动检测
		- 训练数据集构建：by sampling equal numbers of noisy (label-flipped) and clean examples from the original train set
		- 仅使用模型预测的confidence数值这一个特征训练分类器
		- 分类器效果：在类似构建的一个测试集上达到F1=100%；而基于variability的分类器只能达到70%
	- 用这个分类器在全量数据中提取可能的错标数据，给到人工校验
	- 部分数据是确实标注错误，部分数据是ambugious，多人核验的结果不一致


## 总结
- 