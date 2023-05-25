- shows that the pre-trained language models have a low “instrisic dimension” and can still learn effificiently despite a random projection to a smaller subspace.
- Inspired by this, we hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation.

- 官方实现 https://github.com/microsoft/LoRA/tree/main
	- README详细示范了在项目中的使用，非常简单易懂
	- 同时提供了很多训练Transformers模型的示例
		- 其中 NLG/src/model.py 实现了gpt2，没有依赖huggingface，可以对比学习一下这个代码  #TODO 
	- 