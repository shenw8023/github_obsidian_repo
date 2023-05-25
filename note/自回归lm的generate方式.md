
- https://huggingface.co/blog/how-to-generate
- [API文档](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation)

## using different decoding methods
- _auto-regressive_ language generation is based on the assumption that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions:
	 ![[Pasted image 20230509201804.png]]
 
### Greedy Search
- 每步greedily chooses the next word of highest probability
- 容易出现quickly starts repeating itself
- This is a very common problem in language generation in general and seems to be even more so in greedy and beam search
- The major drawback of greedy search though is that <mark style="background: #FF5582A6;">it misses high probability words hidden behind a low probability word</mark>


### Beam search
- Beam search reduces the risk of missing hidden high probability word sequences by keeping the most likely `num_beams` of hypotheses at each time step and eventually choosing the hypothesis that has the <mark style="background: #FF5582A6;">overall highest probability.</mark>
- Beam search will always find an output sequence with higher probability than greedy search, but is not guaranteed to find the most likely output.
```python
# activate beam search and early_stopping
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

```

- 为了缓解仍会出现的重复输出问题，使用 <mark style="background: #FF5582A6;">n-grams (_a.k.a_ word sequences of n words) penalties</mark>
	- The most common _n-grams_ penalty makes sure that no _n-gram_ appears twice by manually setting the probability of next words that could create an already seen _n-gram_ to 0.
	- 一般不常用，会让文本中词汇只能被提到一次，不合理
```python
# set no_repeat_ngram_size to 2
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

```

- <mark style="background: #FF5582A6;">生成前n条beam_search的结果做选择</mark>
```python
# set return_num_sequences > 1
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

# now we have 3 output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

```


### Sampling
- 在开放域文本生成，as humans, we want generated text to surprise us and not to be boring/predictable
- randomly picking the next word according to its conditional probability distribution, 即使是低概率的词也可能被选择
- <mark style="background: #FF5582A6;">注意是根据概率进行采样</mark>
```python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

```

- 问题：The models often generate incoherent gibberish，不通顺的胡说八道
- 一个方案是make the distribution sharper (increasing the likelihood of high probability words and decreasing the likelihood of low probability words) by lowering the so-called `temperature` of the [softmax](https://en.wikipedia.org/wiki/Softmax_function#Smooth_arg_max).
	![[Pasted image 20230509203428.png]]
```python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0, 
    temperature=0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

```

- when setting `temperature` →0, temperature scaled sampling becomes equal to greedy decoding and will suffer from the same problems as before.

### Top-K Sampling
-  the _K_ most likely next words are filtered and the probability mass is redistributed among only those _K_ next words.
	![[Pasted image 20230509203812.png]]

```python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k to 50
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

```

- 问题：One concern though with _Top-K_ sampling is that<mark style="background: #FF5582A6;"> it does not dynamically adapt the number of words</mark> that are filtered from the next word probability distribution 
-  limiting the sample pool to a fixed size _K_ could endanger the model to produce gibberish for sharp distributions and limit the model's creativity for flat distribution.

### Top-p (nucleus) sampling
- Instead of sampling only from the most likely _K_ words, in _Top-p_ sampling chooses from the smallest possible set of words whose cumulative probability exceeds the probability _p_. The probability mass is then redistributed among this set of words.
- This way, the size of the set of words (_a.k.a_ the number of words in the set) can dynamically increase and decrease according to the next word's probability distribution.
![[Pasted image 20230509204100.png]]

```python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

```

- <mark style="background: #FF5582A6;">结合使用top_k, top_p</mark>
	- which can avoid very low ranked words while allowing for some dynamic selection.
```python
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

```


### 其他参数
-  `min_length` can be used to force the model to not produce an EOS token (= not finish the sentence) before `min_length` is reached. This is used quite frequently in summarization, but can be useful in general if the user wants to have longer outputs.
- `repetition_penalty` can be used to penalize words that were already generated or belong to the context. It can be quite effective at preventing repetitions, but seems to be very sensitive to different models and use cases
- `attention_mask` can be used to mask padded tokens
- `pad_token_id`, `bos_token_id`, `eos_token_id`: If the model does not have those tokens by default, the user can manually choose other token ids to represent them.