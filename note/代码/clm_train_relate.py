"""
clm模型在指令微调训练的时候，多轮对话数据的训练形式：

输入数据：
    "conversations":[
        {"from": "human", "value": "题目：小明买了一支钢笔，花费了5元，又买了一本书，花费8元，现在他手里还有10元钱，他手上原来有多少钱？"},
        {"from": "assistant", "value": "\n令小明手上原来有的钱为X元。根据题目描述，得出以下方程式：\nX - 5 - 8 = 10\n化简可得：\nX = 23\n因此，小明手上原来有23元钱。"},
    ]

思路：
    对一条样本:
    输入拼接成:
        [
            human: xxx \n\n Assistand: \n , 
            answer1,
            human: xxx \n\n Assistand: \n , 
            answer2,
            human: xxx \n\n Assistand: \n , 
            answer3,
        ]
   
    join起来就是一条样例的input

    对应的label:
        [
            -100 -100 -100 -100,
            answer1,
            -100 -100 -100 -100,
            answer2,
            -100 -100 -100 -100,
            answer3,

        ]

    join起来就是一条样例的label

    然后在GPT2LMHeadModel forward的时候会对input舍弃最后一个token，最label做right_shift_one，搞定。
    可以看出来多轮对话在训练的时候，单轮也同时被训练了 #[ ]




"""


import copy
IGNORE_INDEX = -100


def generate_and_tokenize_prompt(data_point):
    input_ids = []
    labels = []
    source = data_point["conversations"]
    for sentence in source:
        sentence_from = sentence["from"].lower()
        sentence_value = 'Human: \n' + sentence["value"] + '\n\nAssistant: \n' if sentence_from == 'human' else sentence["value"] #https://github.com/LianjiaTech/BELLE/issues/337
        # conversation += sentence_value
        sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
        label = copy.deepcopy(sentence_ids) if sentence_from != 'human' else [IGNORE_INDEX] * len(sentence_ids)
        input_ids += sentence_ids
        labels += label
        # add eos at every end of assistant sentence
        if sentence_from != 'human':
            input_ids += [tokenizer.eos_token_id]#make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]

    input_ids = input_ids[:training_args.model_max_length-1]
    labels = labels[:training_args.model_max_length-1]
    if not any(x > -100 for x in labels):
        labels[18:24] = input_ids[18:24]#labels can not have all values being -100. 18 and 24 are just random numbers

    attention_mask = [1] * len(input_ids)
    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    return tokenized_full_prompt




# ========================================================

"""
loss计算和困惑度计算

"""
import torch
from torch.nn import CrossEntropyLoss

loss_fct = CrossEntropyLoss()

shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

eval_loss = 0.0
lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
eval_loss += lm_loss.mean().item()
eval_loss = eval_loss / total_eval_steps #计算总平均损失

perplexity = torch.exp(torch.tensor(eval_loss)) #log_perplexity == cross_entropy_loss
