
# RW model

- [参考知乎](https://zhuanlan.zhihu.com/p/610147705)
- deepspeed-chat中是用L维度上的最后一个位置的值当作为本句话的奖励得分
## direct score
- 直接用MSE loss计算正确分数和预测分数的差距
## rank score
- 一般模型都是常规的HF模型
- 奖励模型的输入是prompt+answer的形式
- 如果是BERT类，可以使用pooler_output的输出使用一个linear_layer转为一个标量作为分数
- rank_loss

```python
def rank_loss(rank_rewards_list):
    loss, counts = torch.tensor([0]), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards) - 1):  # 遍历所有前项-后项的得分差
            for j in range(i + 1, len(rank_rewards)):
                diff = nn.functional.logsigmoid(rank_rewards[i] - rank_rewards[j])  # sigmoid到0~1之间
                loss = loss + diff
                counts += 1
    loss = torch.tensor(loss / counts)
    return -loss  # 要最大化分差，所以要取负数

```



# Critic Model
- 