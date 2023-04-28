
# Accelerate

- pip install accelerate
- [官网 github](https://github.com/huggingface/accelerate)
- [Transformers模型训练中结合acclerate教程](https://huggingface.co/docs/transformers/v4.28.1/en/accelerate#distributed-training-with-accelerate) 示例基本上够用了
- [官方文档](https://huggingface.co/docs/accelerate/main/en/index)


## 场景
-  Accelerate abstracts exactly and only the boilerplate code related to **multi-GPUs/TPU/fp16** and leaves the rest of your code unchanged.
- 想对训练循环的细节有足够控制，但是又不想关心分布式相关处理细节
- 如果不想自己写训练循环，完全有更多高层级的抽象框架支持一键分布式训练


## launching script
- 多种启动方式
1. 命令行配置参数 + 脚本启动
   - 执行 `accelerate config` 回答相关问题，获得参数配置文件
     - [选项中涉及用到的新的即时编译工具torch dynamo](https://zhuanlan.zhihu.com/p/592357069)
   - 执行 `accelerate launch my_script.py --args_to_my_script`

2. 直接执行
   - `python my_script.py` or `python -m torchrun my_script.py`
   
3. 结合notebook使用
   - 参考官网github或者[文档](https://huggingface.co/docs/accelerate/v0.18.0/en/package_reference/launchers#accelerate.notebook_launcher)



## Supported 
- CPU only
- multi-CPU on one node (machine)
- multi-CPU on several nodes (machines)
- single GPU
- multi-GPU on one node (machine)
- multi-GPU on several nodes (machines)
- TPU
- FP16 with native AMP (apex on the roadmap)
- DeepSpeed support (Experimental)
- PyTorch Fully Sharded Data Parallel (FSDP) support (Experimental)
- Megatron-LM support (Experimental)




- [相关函数方法用法查阅](https://huggingface.co/docs/accelerate/v0.18.0/en/package_reference/accelerator#accelerate.Accelerator.gather)



# Quicktour

## 基本用法
```python
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()

```


3. 所有训练相关的对象都要通过 prepare() 方法包装一下
    ```
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    ```
- 实际batch_size大小 = 设备数量 * 脚本设置的batch_size
- 或者在实例化 Accelerator(split_batches=True) 使得batch_size总是不变
- lr_scheduler只有需要每次在optimizer step时做step的情况下才需要用prepare()包装 #[ ]
- You may or may not want to send your validation dataloader to prepare(), depending on whether you want to run distributed evaluation or not (see below).


4. Replace the line loss.backward() by accelerator.backward(loss).


## Distributed evaluation
- 不使用分布式evaluation
  - You can perform regular evaluation in your training script, if you leave your validation dataloader out of the prepare() method. In this case, you will need to put the input data on the **accelerator.device** manually.

- 使用分布式
  - `validation_dataloader = accelerator.prepare(validation_dataloader)`
  - 每个设备只能看evaluation data的部分内容，意味着需要使用 `gather_for_metrics()` 方法来group your predictions together
    ```python
    for inputs, targets in validation_dataloader:
        predictions = model(inputs)
        # Gather all predictions and targets
        all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
        # Example of use with a *Datasets.Metric*
        metric.add_batch(all_predictions, all_targets)
    ```

  - Similar to the training dataloader, passing your validation dataloader through prepare() **may change it**: if you run on X GPUs, it will have its length divided by X (since your actual batch size will be multiplied by X), unless you set split_batches=True.
  - Any instruction using your training dataloader length (for instance if you need the number of total training steps to create a learning rate scheduler) should **go after the call to prepare()**.
  - Some data at the end of the dataset may be duplicated so the batch can be divided equally among all workers. As a result, metrics should be calculated through the `gather_for_metrics()` method to automatically **remove the duplicated data while gathering.**
    - 如果不希望自动完成该操作，`gather()` can be used instead to gather the data across all processes and this can manually be done instead.
    - The `gather()` and `gather_for_metrics()` methods require the tensors to be all the same size on each process. If you have tensors of different sizes on each process (for instance when dynamically padding to the maximum length in a batch), you should use the `pad_across_processes()` method to pad you tensor to the biggest size across processes.


## Launching your distributed script
- `accelerate config` 命令行启动
- You can also specify with the flag `--config_file` the location of the file you want to save.
- 使用 `accelerate test` 测试是否一切都没问题
  - 如果前面指定了配置文件地址，这里需要提供 `accelerate test --config_file path_to_config.yaml`
- 启动训练 `accelerate launch --config_file path_to_config.yaml path_to_script.py --args_for_the_script`


## 其他注意事项
- Execute a statement only on one processes
  - for instance a data download or a log statement.To do this, wrap the statement in a test like this:
    ```python
    if accelerator.is_local_main_process:
        # Is executed once per server
    ```
  - progress bars 只在主进程打印一次
    ```python
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    ```

  - 如果是多机多卡，上述方式会在每台机器的主进程执行一次；如果希望在所有进程中只执行一次（而不是每台机器），例如上传最后的模型到model hub；那么使用下面方式包裹：
    ```python
    if accelerator.is_main_process:
        # Is executed once only
    ```

  - 常规的每台机器只执行一次的print操作可以使用`accelerator.print`

## 等待和同步
- 虽然每个进程都是按顺序执行，但是进程之间总有快慢区别， For instance, you shouldn’t save a model before being sure every process is done with training. To do this, just write the following line in your code:
  - `accelerator.wait_for_everyone()`
  - This instruction will block all the processes that arrive first until all the other processes have reached that point (if you run your script on just **one GPU or CPU, this won’t do anything**).


## Saving/loading a model
- 保存模型需要做一些变动
  - first you should wait for all processes to reach that point in the script as shown above
  - then, you should unwrap your model before saving it. 因为prepare() 方法后，你的模型已经被包装在一个更大的模型里面，目的是方便处理分布式训练，如果不去除的话，将导致你的模型参数无法加载到原先的模型
    - unwrap your model
    ```python
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), filename)
    ```


## Saving/loading entire states
- you may want to save the current state of the model, optimizer, random generators, and potentially LR schedulers to be restored in the same script. 
  - You can use `save_state()` and `load_state()` respectively to do so.


## Gradient clipping
- If you are using gradient clipping in your script, you should replace the calls to torch.nn.utils.clip_grad_norm_ or torch.nn.utils.clip_grad_value_ with clipgrad_norm() and clipgrad_value() respectively.


## Mixed Precision training
```python
with accelerator.autocast():
    loss = complex_loss_function(outputs, target):
```

```python
if not accelerator.optimizer_step_was_skipped:
    lr_scheduler.step()
```

## Gradient Accumulation
```python
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

for input, label in training_dataloader:
    with accelerator.accumulate(model):
        predictions = model(input)
        loss = loss_function(predictions, label)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

# HOW-TO-GUIDES
## Start Here 
- 列举了最简单的适用，包括不同launch方式



## How perform inference on large models with small resources




# REFERENCE
- LOGGING