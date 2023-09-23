import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()

        self.multi_head_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, inputs):
        inputs.transpose_(1,2)
        attention_output, _ = self.multi_head_attention(inputs, inputs, inputs)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_norm1(inputs + attention_output)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.layer_norm2(attention_output + ffn_output)

        return ffn_output
    

if __name__ == '__main__':
    # 定义输入数据
    batch_size = 16
    seq_length = 20
    d_model = 64


    # 创建Transformer层实例
    transformer_layer = TransformerLayer(d_model=d_model, num_heads=8, dff=seq_length, dropout_rate=0.1)
    # d_model ====> d_input
    # 创建输入数据
    inputs = torch.randn(batch_size, d_model, seq_length)

    # 获取输出结果
    outputs = transformer_layer(inputs)

    # 打印输入和输出的形状
    print("输入形状：", inputs.shape)
    print("输出形状：", outputs.shape)