- [核心理解](https://blog.csdn.net/u014311125/article/details/109470137)
- https://zhuanlan.zhihu.com/p/113716961
- https://zhuanlan.zhihu.com/p/266023273
- https://zhuanlan.zhihu.com/p/32423092

- 二元交叉熵
	- ![[Pasted image 20230626195325.png]]
- 引入pt
	- ![[Pasted image 20230626195338.png]]
- 二元交叉熵
	- ![[Pasted image 20230626195408.png]]
- $p_t$反映了什么：
	- ![[Pasted image 20230626195558.png]]

	- 可以看到，不管是正样本还是负样本，模型预测正确时​$p_t$都很大，预测错误时$p_t$值很小，所以$1-p_t$能用作对样本预测难易程度的加权
- Focal-loss
	- ![[Pasted image 20230626195939.png]]

	- 图中well-classified examples，不管是正例还是负例，易分类样本（正例总是被预测大于0.5，负例总是被预测小于0.5）其$p_t>0.5$，难分类样本其$p_t<0.5$，$p_t$值越大，表示预测越准确。
	- 不管是难分类还是易分类样本，Focal Loss相对于原始的CE loss都做了衰减，只是难分类样本相对于易分类样本衰减的少。这里的超参数$\gamma$决定了衰减的程度，从上图可以看出$\gamma$越大，损失衰减越明显。
- 超参数理解：
	- ![[Pasted image 20230626201357.png]]
	-  $\alpha$=0.25，$\gamma$=2时精度最高
	- $\alpha$代表了样本数量较少的类的权重，也就是绝大多数情况下的正样本。
	- $\alpha$和$\gamma$是相互作用的，随着$\gamma$的增加，$\alpha$应该稍微降低，主要是配合$\gamma$一起发生作用的



- 多分类下的focal-loss
	- ![[Pasted image 20230626201120.png]]
	- 代码参考bert4torch项目实现
