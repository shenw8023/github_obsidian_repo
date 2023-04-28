- [原文](https://mp.weixin.qq.com/s/UiVYvNUXfW6ycTOC4gCPrg)
- [数据处理相关代码 github](https://github.com/togethercomputer/RedPajama-Data)
- [通过huggingface下载数据集](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
「红睡衣」开源计划总共包括三部分：

  
1. 高质量、大规模、高覆盖度的预训练数据集；
	- 预训练数据集RedPajama-Data-1T已开源
	- 数据预处理相关脚本，分词等均开源
	- 完整的RedPajama-Data-1T数据集需要的存储容量为压缩后3TB，解压后5TB
		![[c424bd1ca438bd8e8e159a0d1ff837ae.png]]

3. 在预训练数据集上训练出的基础模型；
	- 目前开发团队正在橡树岭领导计算设施(OLCF)的支持下开始训练模型，预计几周后即可开源。

5. 指令调优数据集和模型，比基本模型更安全、可靠。
	- 通过OpenChatKit，研究人员已经收到了数十万条高质量的自然用户指令，将用于发布 RedPajama 模型的指令优化版本。
