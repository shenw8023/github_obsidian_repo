- 使用一块24G显存的3090测试完整类chatgpt模型端到端训练过程
- 由于执行脚本中涉及的基础模型最小的都加载不了，所以只能具体为每个步骤创建新的脚本使用更小的模型测试。
- [项目github](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- 项目存放位置：
	- 110.11 容器 038404960ece
	- /data/open_project/DeepSpeedExamples/applications/DeepSpeed-Chat/
	- conda环境：chatglm6b
- 单独训练各步骤
	- step1_supervised_finetuning
		- 修改初始模型使用huggingface gpt2，执行训练脚本：`training_scrips/single_gpu/run_gpt2.sh`
		- 相关数据集自动加载的，也没有下载到本地
		- <mark style="background: #BBFABBA6;">训练完成</mark>，每个epoch会评估ppl，但是日志并没看到<mark style="background: #FF5582A6;">loss相关信息</mark>
		- 测试训练后模型和初始模型表现：`evaluation_scripts/run_prompt.sh`
		- 
	- step2_reward_model_finetuning
		- 修改初始模型使用huggingface distilgpt，执行训练脚本：`training_scripts/single_gpu/run_distilgpt.sh`
		- <mark style="background: #BBFABBA6;">训练完成</mark>，每个epoch会测试acc
		- 测试训练后模型打分能力：`evaluation_scripts/run_eval.sh`
		- 
	- step3_rlhf_finetuning
		- 修改脚本指定相关模型：`training_scripts/single_gpu/run_gpt2.sh`
		- <mark style="background: #FF5582A6;">没有成功</mark>，错误：`RuntimeError: CUDA error: misaligned address` 可能是deepspeed框架问题
- 微调代码：
```python
 for step, batch in enumerate(train_dataloader):
		batch = to_device(batch, device)
		outputs = model(**batch, use_cache=False)
		loss = outputs.loss
		model.backward(loss)
		model.step()
```

## 思考
- 项目优势：端到端，直接加载huggingface仓库的模型作为基础，自动下载一批特定的数据集，遵循 Instruct-gpt 直接执行完整的三个步骤训练。
- 微调的训练逻辑就是huggingface的 casual LM 的训练过程
	- 加载数据
	- 加载模型
	- deepspeed相关设置
	- 训练循环
- 需要先掌握 deepspeed框架的使用
- 模型细节和微调阶段的训练细节这里不涉及，需要看TencentPretrain和huggingface
	-  [[TencentPretrain]]
- RLHF的代码这里是重新写的，需要再整理 #TODO 

