
- ROUGE指标与BLEU指标非常类似，均可用来衡量生成结果和标准结果的匹配程度，不同的是ROUGE基于召回率，BLEU更看重准确率。

## ROUGE-N
- ![[Pasted image 20230627195003.png]]

## ROUGE-L
- L表示: Longest Common Subsequence 最长公共子序列（不需要连续）
- ![[Pasted image 20230627194505.png]]
- ![[Pasted image 20230627194620.png]]

## ROUGE-W
- ![[Pasted image 20230627194804.png]]

- 缺点：
	- 缺点是这种方法只能在单词、短语的角度去衡量两个句子的形似度。并不能支持同义词、近义词等语意级别去衡量。