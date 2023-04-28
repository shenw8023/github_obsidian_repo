- [原文](https://mp.weixin.qq.com/s/Nuxgj7cXwnMEA1dTvpOSVA)
- [github](https://github.com/OpenLMLab/MOSS)
- 
## OpenChat 001
- 从OpenAI的论文附录里扒了一些它们API收集到的user prompt，然后用类似Self-Instruct的思路用text-davinci-003去扩展出大约40万对话数据
	- 跟今天的alpaca类似
	- 多轮对话而不是单轮指令
- 之后在16B基座（CodeGen）上做了一下fine-tune，发现似乎稍微大点的模型很容易学到指令遵循能力
- 一月份的OpenChat 001就已经具备了指令遵循能力和多轮能力，而且我们还惊喜的发现它具有很强的跨语言对齐能力，它的基座预训练语料中几乎不存在中文，但是却<mark style="background: #BBFABBA6;">可以理解中文</mark>并用英文回答

## MOSS 002
- OpenChat 001<mark style="background: #FFB8EBA6;">不具备中文回答能力</mark>，不知道关于自己的信息（比如名字、能力等），且安全性较低
- 一方面加入了约<mark style="background: #BBFABBA6;">30B中文token继续训练基座</mark>，
- 另一方面也加入了大量中英文helpfulness, honesty, harmlessness对话数据，这部分数据共计<mark style="background: #BBFABBA6;">116万条对话</mark>，
	- [开源在huggingface](huggingface.co/datasets/fnlp/moss-002-sft-data)
- 2月21号开放内测，以获取真实用户意图分布。

## MOSS 003
- 开放内测的同时，我们也在继续加大中文语料的预训练，
	- 截止目前MOSS 003的基座语言模型已经在<mark style="background: #BBFABBA6;">100B中文token上</mark>进行了训练，
	- 总训练token数量达到<mark style="background: #BBFABBA6;">700B</mark>，
	- 其中还包含约<mark style="background: #BBFABBA6;">300B代码</mark>。
- 以这部分真实数据作为seed重新生成了约<mark style="background: #BBFABBA6;">110万常规对话数据</mark>，涵盖更细粒度的helpfulness数据和更广泛的harmlessness数据。
- 此外，还构造了<mark style="background: #BBFABBA6;">约30万插件增强的对话数据</mark>，目前已包含搜索引擎、文生图、计算器、方程求解等
	- [开源了一小部分作为示例](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)
	- <mark style="background: #FF5582A6;">这部分示例数据非常值得参考</mark> #TODO 
	- 这部分结构化数据训练的时候以什么形式加载？
- MOSS 003的web端是支持启用哪些插件的
	- 是通过meta instruction来控制，类似gpt-3.5-turbo里的system prompt 
		- 可以在示例的sft_data中看到 [[conversation_with_plugin]]
		- 实际线上部署的时候，应该会根据当前可用插件设定当前模型的meta_instruction
		- 因为是模型控制的所以并不能保证100%控制率，以及还存在一些多选插件时调用不准、插件互相打架的缺陷
- 项目内容
	- 插件版MOSS使用方式将尽快提供
	- 当前三个模型上传到huggingface
		1.  moss-moon-003-base 基座语言模型，具备较为丰富的中文知识。
		2. moss-moon-003-sft 对话微调模型，具备初步的helpfulness, honesty, harmlessness
		3. moss-moon-003-sft-plugin 插件增强的对话微调模型，具备调用至少四种插件的能力
	- Int-4/8量化后的MOSS模型会尽快开放
	- 让MOSS看视频项目：video_chat_with_MOSS
	- 相关前后端代码示例：
		- https://github.com/singularity-s0/openchat_frontend
		- https://github.com/JingYiJun/MOSS_backend