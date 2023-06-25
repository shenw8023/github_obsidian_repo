## chat-gpt和GPT4升级，支持函数调用功能
- [参考](https://mp.weixin.qq.com/s/WMtLojFV5EqV-jurn9WNZQ)
- 以前用户需要在Prompt里加一堆描述才能支持Agent，而现在就简单多了，只要在API请求的时候，传入一个functions参数即可。
- functions参数实际上就类似于用户要指定的Agent，描述了该插件能做什么，要求的入参是什么样的。
- 注意，这里可以是不止一个function，function是一个对象，还可以加上名称（name）、描述（description）、参数（parameters）等等。
- <mark style="background: #FFB8EBA6;">这些模型已经过微调，既能检测到何时需要调用函数（取决于用户输入的内容），又能用符合函数签名的JSON来进行响应。</mark>
- function最大的价值就是解决了GPT返回数据结构化的问题，不再需要用户输入复杂的prompt。

- 具体过程
	- 输入对话内容，和现有的函数信息，gpt会自动判断需不需要调用函数，以及调用哪个函数
		- ![[f19f8eed4e8af066c7b08121dae73a5d.png]]
	- 如果需要调用函数就会输出函数名和入参
		- ![[5f6362fb80206f56ccb200ca7e2ed0f7.png]]

	- 拿着函数名和入参，自己就可以调用第三方插件了，获得结果
		- ![[d7a996715abbe4255234848abe0cdd88.png]]
	- 把结果和上文内容送给gpt，输出整合后的最终回答
		- ![[1bd96442854790a20e0834297e304749.png]]
- 总结：
	- 最大的变化是省略了用户自己在prompt中描述可用函数以及教模型调用函数的的内容