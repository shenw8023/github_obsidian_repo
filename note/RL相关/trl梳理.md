- [参考知乎](https://zhuanlan.zhihu.com/p/606328992)

## 基本概念
- 概念要点：
	- critic model：
		- 使用的是Actor_model的value_head输出的是每个token对应的值。最好使用独立的模型，否则公共部分是否要一起更新也不太明确
		- 表示每个token到结束位置的value
	- reward model
		- 使用的是独立的模型，不做更新。
		- 输出的是整个句子的total Reward：score
		- 会被用于参与计算每个token的即时reward：r
		- 即时reward主要是-kl，但是在最后一个位置是总score
	- ref_model
		- 不是off-policy中提到的真实与环境交互的q分布采样模型
		- 仅仅用于计算KL-penalty，也不做更新
	- <mark style="background: #FFB8EBA6;">off-policy怎么体现，importance-sampling中的q分布怎么来</mark>
		- 与环境交互的模型，实际用的是上一个epoch下的actor，会先把相关的量计算好，在多轮的ppo-minibatch训练过程中，重复使用这些量
		- 所以actor在一个epoch的开始是作为与环境交互reference_model，然后在ppo-minibatch训练时又作为要更新的模型。

	- value和reward是怎么结合的
		- values是模型输出的，每个位置是向后的累积价值
		- rewards是每个位置的即时奖励r，是通过KL和整个句子奖励scores计算得到的
		- 我们的目标是计算每个位置的$advantage_t = r_t + V_{t+1} - V_t$
		- advantage表示的每个位置的向后的相对累积价值，相对体现在减掉了价值的期望

- 公式
	- ![[Pasted image 20230607163547.png]]


		![[Pasted image 20230607163602.png]]



## PPO计算流程
- 通过actor_mode计算每个位置的 logprobs 和 values
- 通过ref_model计算每个位置的 ref_logprobs
- 通过reward_model计算每句话的总分 scores

- 计算每个位置的即时rewards:r
	- 用到 logprobs和ref_logprobs计算KL
	- 将scores加到最后一个位置

- train-minibatch-ppo
	- 计算pg_loss
		- 利用rewards和values计算advantage
		- advantage.detach
		- 重新计算当前正在训练的actor得到当前步的logprobs和values
		- importance-sampling得到ratio，用到的old_logprobs是流程一开始得到的logprobs
		- 得到pg_loss

	- 计算value_loss
		- 当前的 (values - (advantage + old_values))**2
			- MSE loss
			- 这里用到的old_values也是流程一开始得到的values

  
  

## 整体流程
- 每个epoch：
	- 将当前的actor视为未来一段实际内ppo-minibatch训练期间实际与环境交互的模型，
	- 采样128个prompt作为query，用actor生成128个response，计算log_prob等信息
		- 每个ppo-epoch：
			- 每次使用一个完整句子的相关信息，做minibatch-ppo训练，更新actor参数
			- 在每次迭代计算的时候，importance-sampling用到的old_logprob，就是上面的log_prob，是不变的
		- 结束本轮ppo迭代后，当前actor的参数被更新
	- 当前actor已经被更新，继续做epoch时，相当于真正的跟环境交互的模型也更新了。

  
  
## 疑问

- value_loss的计算公式不太明白
	- value_loss 就应该等于critic产生的预测值 v_pred 和真实值 r + v_next 之间的差值
	- 这个真实值怎么理解，它不也就是上一轮中的critic预测的值吗这里的v_next也是一个变量呀，难度是因为用r做了修正，我们希望更新critic，使得他的输出更符合v = r + v_next


- 计算每个位置的reward，为什么最后一个位置是句子的score，其他位置是KL-penalty
	- 也许应该去TD的过程找到原因