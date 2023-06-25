- [表示学习中7大损失函数]()
## Triplet loss
- [参考zhihu](https://zhuanlan.zhihu.com/p/171627918)
- 形式：$$L=max(d(a,p)-d(a,n)+margin, 0)$$
- 输入是一个三元组，包括锚（Anchor）示例、正（Positive）示例、负（Negative）示例，通过优化锚示例与正示例的距离小于锚示例与负示例的距离，实现样本之间的相似性计算。
- 样本可以分为三类：
	- easy triplets : $L=0$，即$d(a,p)+margin<d(a,n)$
	- hard triplets : $L > margin$，即$d(a,n) < d(a,p)$ 
	- semi-hard triplets : $L < margin$，即$d(a,p)<d(a,n)<d(a,p)+margin$
- 为什么要设置margin？
	- <mark style="background: #FFB8EBA6;">核心：不仅要让a到p的距离小于a到n的距离，而且要小至少一个margin的距离才不计入loss，也就是说差距要够大到一定程度才行。</mark>
	- 避免模型走捷径，将negative和positive的embedding训练成很相近，因为如果没margin，triplets loss公式就变成了 $L=max(d(a,p)-d(a,n),0)$ ，那么只要$d(a,p)=d(a,n)$ 就可以满足上式，也就是锚点a和正例p与锚点a和负例n的距离一样即可，这样模型很难正确区分正例和负例。
	- 设定一个margin常量，可以迫使模型努力学习，能让锚点a和负例n的distance值更大，同时让锚点a和正例p的distance值更小。
	- 由于margin的存在，使得triplets loss多了一个参数，margin的大小需要调参。如果margin太大，则模型的损失会很大，而且学习到最后，loss也很难趋近于0，甚至导致网络不收敛，但是可以较有把握的区分较为相似的样本，即a和p更好区分；如果margin太小，loss很容易趋近于0，模型很好训练，但是较难区分a和p。

- 

## InfoNCE loss
