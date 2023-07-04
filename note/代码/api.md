- 多标签分类
```python
logits = torch.randn(3,4)
outputs = torch.sigmoid(logits).detach().cpu().numpy()
outputs = (outputs > 0.5).astype(int)
outputs
```


- 单分类
```python
outputs = torch.max(logits.data, 1)[1].cpu().numpy()
preds.extend(outputs.tolist())
```

torch.softmax
torch.nn.functional.softmax

torch.sigmoid
torch.nn.functional.sigmoid 要被删了，别用

torch.argmax
Tensor.argmax

