from transformers import BertModel, BertConfig, BertPreTrainedModel 
from torch import nn
import torch


class MyModel(BertPreTrainedModel):
    def __init__(self, config:BertConfig):
        super().__init__(config)
    
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# training
model = MyModel.from_pretrained('bert-base-chinese')  #默认实例化一个BertConfig传入
model.save_pretrained(save_path)

# deploying
model = MyModel.from_pretrained(save_path)

model_state_dict = torch.load(save_path+"pytorch_model.bin")
model = MyModel.from_pretrained('bert-base-chinese', state_dict=model_state_dict)



# training standard
config = BertConfig.from_pretrained(args.bert_config_path)
config.num_p = 10 #一些参数的定制化
model = MyModel.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)



# lr_scheduler
from transformers import get_linear_schedule_with_warmup, AdamW

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in train_model.named_parameters() if "bert." in n],
        "weight_decay": args.weight_decay,
        "lr": args.bert_learning_rate,
    },
    {
        "params": [p for n, p in train_model.named_parameters() if "bert." not in n],
        "weight_decay": args.weight_decay,
        "lr": args.other_learning_rate,
    }
]

optimizer = AdamW(optimizer_grouped_parameters, eps=args.min_num)

t_total = len(dataloader) * args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )


# EpochRunner不需要重复接收各个组件，直接将组件传给实际执行的StepRunner，实例化后传给EpochRunner



# 如果分数上升，保存模型，并记下保存路径；如果分数下降，加载上一个模型的保存路径，并降低学习率为一半（重新初始化优化器，清空动量信息，而不是只修改学习率----使用PyTorch的话新建一个新优化器即可）