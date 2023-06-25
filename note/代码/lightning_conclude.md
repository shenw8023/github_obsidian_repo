- https://zhuanlan.zhihu.com/p/319810661
- https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#val-check-interval

## log()方法
- 追踪记录指标

### 两种使用方式
- 单个指标记录
    ```
    value=10
    self.log("train_loss", value)
    ```
- 涉及多个指标要记录时
    ```
    values = {"loss": loss, "acc": acc}  
    self.log_dict(values)
    ```

### 其他相关参数
- 在命令行进度条中显示指标
    - prog_bar=True
    - e.g. `self.log(..., prog_bar=True)`
- 在浏览器中显示指标
    - 默认情况下，lightning已经使用了Tensorboard这个experiment manager
    - 使用 `tensorboard --logdir=lightning_logs/` 开启tensorboard

- 汇聚指标数值 #TODO
    - 在training_step中调用self.log，产生的是每个step的指标结果
    - 在validation_step和test_step中调用self.log，默认会自动计算为所有step的数值的均值
        ```python
        def validation_step(self, batch, batch_idx):
            value = batch_idx + 1
            self.log("average_value", value)
        ```
- 指标保存路径
    - 默认所有logged的指标都会被保存在当前工作路径，使用Trainer中的default_root_dir做修改
    - `Trainer(default_root_dir="/path/") `

    

### 参数
- on_step
    - if True logs at this step
    - 如果是在traning_step中调用log()，该参数默认为True，而在validation/test_step中调用log()时默认为False

- on_epoch
    - if True, logs epoch accumulated metrics
    - 在training_step中该参数默认为False，在validation/test_step中该参数默认为True
    - 也就是说如果在

- reduct_fx(Union[str, Callable]) 
    - reduction function over step values for end of epoch. torch.mean() by default.
    - 可选: max, min, mean


### log frequency
- It may slow down training to log on every single batch. By default, Lightning logs every 50 rows, or 50 training steps. To change this behaviour, set the log_every_n_steps Trainer flag.
- `trainer = Trainer(log_every_n_steps=10)`





## Trainer参数理解

- **basic use**
    ```python
    from argparse import ArgumentParser

    def main(hparams):
        model = LightningModule()
        trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
        trainer.fit(model)

    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--accelerator", default=None)
        parser.add_argument("--devices", default=None)
        args = parser.parse_args()
        main(args)

    >>> python main.py --accelerator 'gpu' --devices 2

    ```
- **自动解析命令行参数到Trainer中**
    - Pro-tip: You don’t need to define all flags manually. Lightning can add them automatically
        ```python
        from argparse import ArgumentParser

        def main(args):
            model = LightningModule()
            trainer = Trainer.from_argparse_args(args)
            trainer.fit(model)

        if __name__ == "__main__":
            parser = ArgumentParser()
            parser = Trainer.add_argparse_args(parser)
            args = parser.parse_args()
            main(args)
        >>> python main.py --accelerator 'gpu' --devices 2 --max_steps 10 --limit_train_batches 10 --any_trainer_arg x
        ```

- 捕捉Ctrl+C，自动调用callback的on_train_end方法，优雅处理

- 训练前或训练后单独执行一次validation epoch
    `trainer.validate(model=model, dataloaders=val_dataloaders)`

- 训练后测试
    `trainer.test(dataloaders=test_dataloaders)`

- 可复现性
  - 需要设置随机数种子
  - 设置deterministic=True in Trainer
    ```python
    from pytorch_lightning import Trainer, seed_everything
    seed_everything(42, workers=True) # sets seeds for numpy, torch and python.random.
    model = Model()
    trainer = Trainer(deterministic=True) #defalt=False，可能会降低系统速度
    ```

- **accelerator**：加速器
  - 相关参数
    ```python
    # CPU accelerator
    trainer = Trainer(accelerator="cpu")

    # Training with GPU Accelerator using 2 GPUs
    trainer = Trainer(devices=2, accelerator="gpu")

    # Training with TPU Accelerator using 8 tpu cores
    trainer = Trainer(devices=8, accelerator="tpu")

    # Training with GPU Accelerator using the DistributedDataParallel strategy
    trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp")

    # 自动检测GPU加速器
    trainer = Trainer(devices=2, accelerator="auto")
    ```

  - 如果没有指定`devices`参数
    ```python
    # Training with GPU Accelerator using total number of gpus available on the system
    Trainer(accelerator="gpu")
    ```

- **devices**：指定设备数量或者序号
  - Number of devices to train on (int), which devices to train on (list or str), or "auto"
    ```python
    # Training with CPU Accelerator using 2 processes #TODO使用cpu，使用两个进程模拟分布式训练，不会提高速度，因为torch本身已经用上了多核。
    trainer = Trainer(devices=2, accelerator="cpu") 

    # Training with GPU Accelerator using GPUs 1 and 3
    trainer = Trainer(devices=[1, 3], accelerator="gpu")

    # Training with TPU Accelerator using 8 tpu cores
    trainer = Trainer(devices=8, accelerator="tpu")

    ```
  - 没有指定时，默认为'auto'，是根据使用的accelerator来决定的
    ```python
    # If your machine has GPUs, it will use all the available GPUs for training
    trainer = Trainer(devices="auto", accelerator="auto")

    # Training with CPU Accelerator using 1 process
    trainer = Trainer(devices="auto", accelerator="cpu")
    ```


- **accumulate_grad_batches**：梯度累积 
  - 默认没有梯度累积
    `trainer = Trainer(accumulate_grad_batches=1)`

  - 两种设定方式
    ```python
    # accumulate every 4 batches (effective batch size is batch*4)
    trainer = Trainer(accumulate_grad_batches=4)

    # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
    trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})
    ```

- auto_scale_batch_size：自动检测最大batch_size

- auto_lr_find：自动寻找最合适学习率
  
- benchmark：提高固定输入维度下的训练速度

- **check_val_every_n_epoch**：按epoch修改validation频率
    ```python
    # default used by the Trainer
    trainer = Trainer(check_val_every_n_epoch=1)

    # run val loop every 10 training epochs
    trainer = Trainer(check_val_every_n_epoch=10)
    ```

- **default_root_dir**：默认存储路径
  - Default path for logs and weights when no logger


- enable_checkpointing
  - 默认会在每个epoch后保存最新的模型参数，不考虑提升与否，且只保留一个
    ```python
    # default used by Trainer, saves the most recent model to a single checkpoint after each epoch
    trainer = Trainer(enable_checkpointing=True)

    # turn off automatic checkpointing
    trainer = Trainer(enable_checkpointing=False)
    ```

- fast_dev_run：验证训练逻辑代码有没有问题
  - 使用n（如果指定的话）或1（如果为True）个batch运行，验证代码正确性
  - 区别于limit_{train,val,test,predict}_batches，该设置会自动删减掉很多不必要的内容，因此不能单纯作为减少数据量的操作来用，仅用于debug。
    ```python
    # default used by the Trainer
    trainer = Trainer(fast_dev_run=False)

    # runs only 1 training and 1 validation batch and the program ends
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(...)

    # runs 7 predict batches and program ends
    trainer = Trainer(fast_dev_run=7)
    trainer.predict(...)
    ```

- gradient_clip_val：梯度裁剪值
    ```python
    # default used by the Trainer
    trainer = Trainer(gradient_clip_val=0.0)
    ```

- **limit_train_batches**：指定使用的train-dataset batch数量
  - 如果要用少批数据完整跑一遍流程，使用该参数
    ```python
    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

    # run through only 25% of the training set each epoch
    trainer = Trainer(limit_train_batches=0.25)

    # run through only 10 batches of the training set each epoch
    trainer = Trainer(limit_train_batches=10)
    ```
- limit_val_batches

- limit_test_batches

- **log_every_n_steps**
  - 指定log()函数写入logger的频率，默认为50个step写入一次，一般都需要修改小一点
  - 跟progress_bar显示的内容没联系

- max_epochs：最大轮数
  - 如果没有指定max_epochs和max_steps，则max_epochs默认设置为1000
  - 要进行无限训练，设置max_epochs=-1

- min_epochs：默认为1

- max_steps: 默认为-1
  - Stop training after this number of `global_steps`. Training will stop if max_steps or max_epochs have reached (earliest).

- min_steps

- max_time：训练的最长时间
    ```python
    # Default (disabled)
    trainer = Trainer(max_time=None)

    # Stop after 12 hours of training or when reaching 10 epochs (string)
    trainer = Trainer(max_time="00:12:00:00", max_epochs=10)

    # Stop after 1 day and 5 hours (dict)
    trainer = Trainer(max_time={"days": 1, "hours": 5})

    ```

- num_nodes=1：分布式训练指定GPU nodes

- **precision**：精度
  - Lightning supports either double (64), float (32), bfloat16 (bf16), or half (16) precision training.
    ```python
    # default used by the Trainer
    trainer = Trainer(precision=32)

    # 16-bit precision
    trainer = Trainer(precision=16, accelerator="gpu", devices=1)  # works only on CUDA

    # bfloat16 precision
    trainer = Trainer(precision="bf16")

    # 64-bit precision
    trainer = Trainer(precision=64)
    ```

- reload_dataloaders_every_n_epochs：动态改变数据源
    - Set to a positive integer to reload dataloaders every n epochs from your currently used data source. DataSource can be a LightningModule or a LightningDataModule.

- **resume_from_checkpoint**：指定checkpoint路径，继续训练
  - 将被弃用，使用`trainer.fit(ckpt_path="some/path/to/my_checkpoint.ckpt") `instead.
    ```python
    # default used by the Trainer
    trainer = Trainer(resume_from_checkpoint=None)

    # resume from a specific checkpoint
    trainer = Trainer(resume_from_checkpoint="some/path/to/my_checkpoint.ckpt")
    ```

- strategy：分布式策略
    ```python
    # Training with the DistributedDataParallel strategy on 4 GPUs
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with the DDP Spawn strategy using 4 cpu processes
    trainer = Trainer(strategy="ddp_spawn", accelerator="cpu", devices=4)
    ```

- **val_check_interval**：validate频率，默认是每个epoch之后一次
    ```python
    # default used by the Trainer
    trainer = Trainer(val_check_interval=1.0)

    # check validation set 4 times during a training epoch
    trainer = Trainer(val_check_interval=0.25)

    # check validation set every 1000 training batches in the current epoch
    trainer = Trainer(val_check_interval=1000)

    # check validation set every 1000 training batches across complete epochs or during iteration-based training
    # use this when using iterableDataset and your dataset has no length
    # (ie: production cases with streaming data)
    trainer = Trainer(val_check_interval=1000, check_val_every_n_epoch=None)
    ```


## 设置checkpoint保存 
- https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing_intermediate.html
- 以下出现的参数可以查看具体api文档

- **save checkpoint by confition**
  - when
    - 如果是基于iterative的训练，没有epoch的话，设置`every_n_train_steps=N.`指定保存时机
    - 每隔多少个epoch（默认为 1）进行保存：指定`every_n_epochs`
    - 每隔多长时间进行保存：指定`train_time_interval`
    - 如果monitor的是training metric，建议指定`save_on_train_epoch_end=True`，确保该指标被正确accumulated
  - which
    - 指定`save_last`，在每轮训练结束后保存一次
      - 会额外维护一个名为last.ckpt的最新保存文件
    - 根据指定的monitor保存top_k个结果
        ```python
        from pytorch_lightning.callbacks import ModelCheckpoint
        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            dirpath="my/path/",
            filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        )

        # saves last-K checkpoints based on "global_step" metric
        # make sure you log it inside your LightningModule
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="global_step",
            mode="max",
            dirpath="my/path/",
            filename="sample-mnist-{epoch:02d}-{global_step}",
        )

        trainer = Trainer(callbacks=[checkpoint_callback])
        ```
    - what
      - 默认保存model weights, optimizer states, etc.
      - 可以指定`save_weights_only=True`只保存模型参数

    - where
      - 通过dirpath指定保存路径
      - 通过filename指定文件名，可以动态注入变量