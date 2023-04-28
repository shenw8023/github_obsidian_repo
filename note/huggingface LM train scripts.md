- /examples/pytorch/language-modeling/run_clm_no_trainer.py
- 依赖库
	- datasets：加载hub中public数据集或本地文件数据
	- accelerate：多卡分布式训练框架，解决to_device等操作

## 主要内容

- parse_args()
- accelerator = Accelerator()
- 设置logging 和每台机器的主进程关联
- 加载数据集：dataset_name or local_file
- config = AutoConfig.from_pretrained(args.model_name_or_path)
- tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,)
- model = AutoModelForCausalLM.from_pretrained()
- dataset多进程处理，tokenize和concat
- dataloader
- 
```
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
```

- train
- eval
## TODO
- accelerate库的基本使用
- 关于多卡训练中batch_size的计算
	- gradient_accumulation_steps设置后的使用
- AutoModelForCausalLM 的output包含loss这个属性，可以直接backward
- huggingface库包含的一些实用模块
	- get_scheduler()
	- default_data_collator()
	- set_seed()
- [[GPT2LMHeadModel]]
- 