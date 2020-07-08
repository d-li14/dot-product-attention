import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, num_head):
        super(DoubleAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_head = num_head

        self.key = nn.Conv2d(in_channels, key_channels, 1)
        self.query = nn.Conv2d(in_channels, key_channels, 1)
        self.value = nn.Conv2d(in_channels, value_channels, 1)
        self.expansion = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        key = self.key(x).reshape((n, self.key_channels, h * w))
        query = self.query(x).reshape(n, self.key_channels, h * w)
        value = self.value(x).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.num_head
        head_value_channels = self.value_channels // self.num_head
        
        attention = []
        for i in range(self.num_head):
            _key = F.softmax(key[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            _query = F.softmax(query[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            _value = value[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            _attention = (_value @ _key.transpose(1, 2) @ _query).reshape(n, head_value_channels, h, w)
            attention.append(_attention)

        x += self.expansion(torch.cat(attention, dim=1))

        return x