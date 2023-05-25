- 用于解决什么样的问题： ^e06760
	- problems where the <mark style="background: #FF5582A6;">output dictionary size</mark> depends on the <mark style="background: #FF5582A6;">number of elements in the input sequence.</mark>
	- 例如，需要decoder输出的是index，对应输入的序列的某项
- 传统的seq2seq，由于decoder的输出是输入序列的index，所以输出dictionary size如果要求跟着输入序列长度变化的情况，就没法用一个模型解决。
- In this sequence-to-sequence model, the output dictionary size for all symbols Ci is fixed and equal to n, since the outputs are chosen from the input. Thus, we need to train a separate model for each n. This prevents us from learning solutions to problems that have an output dictionary with a size that depends on the input sequence length.

- 传统attention方法：
	- ![[Pasted image 20230525185310.png]]
	- $e_j  和 d_i$ 分别表示第j个输入token和第i和输出token，$a$是权重向量
	- Lastly, $d_i^,$ and $d_i$ are <mark style="background: #FF5582A6;">concatenated</mark> and used as the hidden states from which we make predictions and which we feed to the next time step in the recurrent model.

- 本论文方法：
	- 直接将attention后的权重作为当前指针的概率，概率最大的输入token就是当前输出的index
	- ![[Pasted image 20230525185245.png]]
- 输出下一个token的时候，是将当前指针指向的输入token复制过来作为上文，继续做decode