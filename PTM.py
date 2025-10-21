# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.requires_grad = False
        pe = pe.unsqueeze(0)  # 在批次维度上增加维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  # 对位置编码进行广播并添加到输入张量上


class Transformer_encoder(nn.Module):
    def __init__(self):
        super(Transformer_encoder, self).__init__()

        self.Feat_embedding = nn.Linear(1, 512, bias=False) # equal to nn.embedding

        self.pos = PositionalEncoding(512,max_len=100)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                                        nhead=8,
                                                        dim_feedforward=2048,
                                                        batch_first=True,
                                                        dropout=0.1,
                                                        activation="gelu")

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)


        self.feat_map = nn.Linear(20*512, 1, bias=True)

        self.out_fc = nn.Linear(512*20, 1, bias=True)
        self.activation = nn.ReLU()  # 添加激活函数
        self.dropout = nn.Dropout(0.1)  # 添加Dropout层
        self.bn = nn.BatchNorm1d(20)

        """only for transfer learning, please do not compile this code """

        """if you wanna a full fine tuning,just run as pretraining.
            for selective fine tuning, run following code-1 """
        """1.refreeze the last layer """
        for param in self.parameters():
            param.requires_grad = False

        # 解冻最后的全连接层
        self.out_fc.weight.requires_grad = True
        self.out_fc.bias.requires_grad = True

        """2.refreeze the first and last layer """
        # # 解冻最后的全连接层和Feat_embedding
        # for param in self.parameters():
        #     param.requires_grad = True
        #
        # # 冻结transformer_encoder的参数
        # for param in self.transformer_encoder.parameters():
        #     param.requires_grad = False

        """  """

    def forward(self, src, use_dropout=False):
        B, _ = src.size()

        # Embedding
        embedding_src = self.Feat_embedding(src.unsqueeze(2))  # (128,20,1)--(128,20,512)

        embed_encoder_input = self.pos(embedding_src)  # Adding positional encoding

        # Transformer Encoder
        out = self.transformer_encoder(embed_encoder_input)  # (128,20,512)

        x = self.bn(out)
        x = self.activation(x)

        # Apply dropout conditionally
        if use_dropout:
            # Generate a random dropout probability between 0.1 and 0.9
            p = torch.rand(1).item() * 0.8 + 0.1
            x = F.dropout(x, p=p,
                          training=True)  # Dropout during inference with random p # Dropout during inference if requested
        else:
            x = self.dropout(x)  # Standard dropout during training

        # # x = self.dropout(x)
        # if use_dropout:
        #     x = F.dropout(x, p=0.5, training=True)  # Dropout during inference if requested

        x = self.out_fc(x.reshape(B, -1))  # Output layer

        return x


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)