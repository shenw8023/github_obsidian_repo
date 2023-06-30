from torch import nn


# 这里选取了Transformer模型的部分代码
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        # inputs shape: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        # output shape: [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual)
