import torch
from torch import nn

"""
#TODO 梳理一下每个层的实际输入输出，干了什么事，更高的抽象
BertLayer: 

"""


BertLayerNorm = torch.nn.LayerNorm #TODO
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        """
        inputs_embeds: 可以看作是已经查过word_embeddings的输入张量，[B,S,H]
                     我们使用input_ids目标也是获得inputs_embeds
        
        """
        if input_ids is not None:
            input_shape = input_ids.shape  # [B,S]
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device) #TODO绝对位置编码吗？
            position_ids = position_ids.unsqueze(0).expand(input_shape)  # [B,S]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device) # [B,S]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(BertLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = () # 共13个，其中包括输入的hidden_states作为第一个
        all_attentions = () # 共12个，每一个为每层BertLayer计算出的attention
        for i, layer_module in self.layer:
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
    
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        output = (hidden_states)
        if self.output_hidden_states:
            output = output + (all_hidden_states,)
        if self.output_attentions:
            output = output + (all_attentions,)
        return output  # last-layer hidden state, (all hidden states), (all attentions)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder

        # 如果是decoder的话，实际有三个层
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)  

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        # --> hidden_states [B,S,H] + attention_probs [B,num_heads,S,S]
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        #TODO 需要看一下图
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        # --> [B,S,H], [B,num_heads,S,S]
        return outputs


class BertAttention(nn.Module):
    """
    合并BertSelfAttention和BertSelfOutput
    主要进行prune_heads操作
    """
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads:list):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size) # 这里这俩参数必须从实例对象获取，而不能从config，因为他们是跟着prune实际变化的，我们在prune完也会更新他们
        heads = set(heads) - self.pruned_heads #排除已经prune过的heads
        for head in heads:
            #需要先考虑如果是再次对已经prune过的层做prune，要将指定的head_num转为在当前剩下的heads中，从小到大排第几
            head = head - sum(1 if h<head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)  #这里转为一维，因为是要直接作用到SelfAttention的 key,query,value矩阵的，最终体现在让all_head_size对应的部分entries被删掉（index_select）
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        # BertSelfAttention层会在BERT模型初始化的时候做prune_heads，而在forward过程中会用到以下超参数，因此必须相应做修改
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states) #residual connection
        outputs = (attention_output,) + self_outputs[1:]
        return outputs  # 输出 hidden_states [B,S,H] + attention_probs [B,num_heads,S,S]


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        


class BertSelfOutput(nn.Module):
    """
    接attention层之后，作用是对attention输出缩回原来大小，并且add&norm
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) #它的权重会在prune_head中被修改，那样其infeature会小于config.hidden_size
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states [B,S,H]: self_attention层的输出
        input_tensor [B,S,H]: self_attention层的输入
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



## ===============全连接层 + add&norm ===========================

class BertIntermediate(nn.Module):
    """全连接层，输入为：attention block 经过add&norm后的hidden_states
        会将隐层的维度放大
        input: [B,S,H]
        output: [B,S,intermediate_size]    
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

class BertOutput(nn.Module):
    """
    接全连接层后，作用是将全连接层输出缩回原来的大小，并且add&norm
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states [B,S,intermediate_size]: 全连接层的输出
        input_tensor [B,S,H]: 是全连接层的输入，也就是self_attention层的输出——attention_output：
        output: [B,S,H]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



## ==




## =======================utils =====================================

def prune_linear_layer(layer, index, dim=0):
    """
    将线性层的输出维度改为len(index)，to keep only entries in index
    相应网络的weight和bias也会根据index只保留相应的部分
    used to remove heads
    dim指定的是要prune的weight的维度，如果为0：输出维度会变，如果是1：输入维度会变
    """
    # 通过dim和index对权重进行prune
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].close().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    
    # 创建新的网络层
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    # 将prune后的权重替换进去
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous)
        new_layer.bias.requires_grad = True
    return new_layer
