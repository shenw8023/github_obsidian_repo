
import torch
from torch import nn
import math

class BertSelfAttention(nn.Module):
    def __init__(self, config):  # config.hidden_size, config.num_attention_heads, attention_probs_dropout_prob
        super().__init__()
        if config.hidden_size % config.num_attention_heads !=0:
            raise ValueError(
                "the hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            ) 

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x): # x: [B,S,H]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) #[B,S,num_heads,head_size]
        return x.permute(0, 2, 1, 3) #[B,num_heads,S,head_size]


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,   
    ):  

        mixed_query_layer = self.query(hidden_states)  # [B,S,H]
        mixed_key_layer = self.key(hidden_states) 
        mixed_value_layer = self.value(hidden_states)
        
        # 拆分为多头
        query_layer = self.transpose_for_scores(mixed_query_layer) #[B,num_heads, S, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # query * key 计算得分矩阵
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask  #TODO 这里attention_mask形状为[B,1,S,S],在mask的位置值应该为-float('inf')，，其他位置为0，这样在对权重矩阵按行进行softmax归一化的时候这些位置成为0

        # 权重矩阵
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [B, num_heads S, S]
        attention_probs = self.dropout(attention_probs)

        # mask heads
        if head_mask is not None:
            attention_probs = attention_probs * head_mask #TODO head_mask应该是 [B, num_heads,1,1]形状，并且mask的位置值为0

        context_layer = torch.matmul(attention_probs, value_layer) # [B,num_heads,S,S], [B,num_heads,S,head_size] -> [B,num_heads,S,head_size]
        
        # 合并多头，下面两种操作经验证是一样的结果
        # context_layer = context_layer.permute(0,2,1,3).contiguous() #[B,S,num_heads,head_size]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = context_layer.permute(0,2,1,3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer



## =========== 关于attention mask=========

def process_encoder_attention_mask(attention_mask):
    "encoder的padding mask是怎么起作用的"
    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)

    # if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length] 
    # TODO 用于在selfAttention的时候和得分矩阵直接相加
    if attention_mask.dim() == 3: #[B,S,S]
        extended_attention_mask = attention_mask[:, None, :, :] # [B,1,S,S]
    elif attention_mask.dim() == 2: #[B,S] e.g [1,1,1,1,1,0,0]
        extended_attention_mask = attention_mask[:, None, None, :] #[B,1,S,S]
    else:
        raise ValueError

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    ## 最终shape为[B,1,S,S]，第二个维度广播后对应num_heads维度，其中被padding的位置为-10000，其他位置为0，
    ## 在attention_scores位置和其逐元素相加，然后softmax


# =================head mask =========================