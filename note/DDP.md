- https://blog.csdn.net/weixin_44966641/article/details/120248357




### 基本概念

- 两台机器，每台8张卡
在16张显卡，16的并行数下，DDP会同时启动16个进程。下面介绍一些分布式的概念。

**group**

即进程组。默认情况下，只有一个组。这个可以先不管，一直用默认的就行。

**world size**

表示全局的并行数，简单来讲，就是2x8=16。

```python3
# 获取world size，在不同进程里都是一样的，得到16
torch.distributed.get_world_size()
```

**rank**

表现当前进程的序号，用于进程间通讯。对于16的world sizel来说，就是0,1,2,…,15。  
注意：rank=0的进程就是master进程。

```text
# 获取rank，每个进程都有自己的序号，各不相同
torch.distributed.get_rank()
```

**local_rank**

又一个序号。这是每台机子上的进程的序号。机器一上有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7

```python
# 获取local_rank。一般情况下，你需要用这个local_rank来手动设置当前模型是跑在当前机器的哪块GPU上面的。
torch.distributed.local_rank()
```