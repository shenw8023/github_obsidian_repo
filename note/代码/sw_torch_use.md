## torch.Tensor.expand()
- 将一个张量大小为1的维度进行复制扩展到指定大小；
- 如果某个维度指定的数字为-1，表示对应的那个维度不变；
- 可以指定的维数超过原来张量的维数，新的维度总会被设在前面
- 返回的是原张量的一个视图，如果涉及对张量的原地操作，最好先clone一下再操作
```
a = torch.randn(1,3)
b = a.expand(3,3) # 复制扩展了0维
c = a.expand(2,3,3) # 新增的维度总会在前面

```

## torch.index_select
- https://pytorch.org/docs/stable/generated/torch.index_select.html#torch.index_select
- 在指定维度上，根据一个索引张量来取数
- parameters:
  - input (Tensor) – the input tensor.
  - dim (int) – the dimension in which we index
  - index (IntTensor or LongTensor) – the 1-D tensor containing the indices to index

```
x = torch.randn(3, 4)
indices = torch.tensor([0, 2]) 
torch.index_select(x, 0, indices) 
torch.index_select(x, 1, indices)

```
## 一维的张量可以使用索引张量取部分值
```
a = torch.tensor([1,2,3,4,5])
index = torch.tensor([2,3,4])
a[index]
```


## nn.Linear() 的weight参数shape: [outfeature, infeature]
- 计算公式中会对weight做转置：y = xA^T + b


## 关于对叶子节点不能进行in-place操作
- PyTorch doesn’t allow in-place operations on leaf variables that have requires_grad=True (such as parameters of your model) because the developers could not decide how such an operation should behave. If you want the operation to be differentiable, you can work around the limitation by cloning the leaf variable (or use a non-inplace version of the operator).

```
new_layer.weight.requires_grad = False 
new_layer.weight.copy_(W.contiguous())
new_layer.weight.requires_grad = True   
```









# TODOs
- head_mask和attention_mask是怎么参与计算的