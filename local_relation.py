import torch
import torch.nn as nn
from torch.nn import functional as F


class GeometricPrior(nn.Module):
    def __init__(self, k, channels, reduction=0.5):
        super(GeometricPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.l1 = nn.Conv2d(2, int(reduction * channels), 1)
        self.l2 = nn.Conv2d(int(reduction * channels), channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self):
        x_range = torch.arange(-(self.k // 2), self.k // 2 + 1).view(1,-1)
        x_position = x_range.expand(self.k, x_range.shape[1])
        y_range = torch.arange(self.k // 2, -(self.k // 2) - 1, -1).view(-1,1)
        y_position = y_range.expand(y_range.shape[0], self.k)
        position = torch.cat((x_position.unsqueeze(0), y_position.unsqueeze(0)), dim=0).unsqueeze(0).float()
        if torch.cuda.is_available():
            position = position.cuda()
        out = self.l2(self.relu(self.l1(position)))
        return out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])


class KeyQueryMap(nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = nn.Conv2d(channels, channels // m, 1)
    
    def forward(self, x):
        return self.l(x)


class AppearanceComposability(nn.Module):
    def __init__(self, kernel_size, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size, 1, padding, stride)

    def forward(self, x):
        key_map, query_map = x # (N, C/m, H, W)
        key_map_unfold = self.unfold(key_map).transpose(2, 1).contiguous() # (N, H/sxW/s, C/mxkxk)
        query_map_unfold = self.unfold(query_map).transpose(2, 1).contiguous() # (N, H/sxW/s, C/mxkxk)
        key_map_unfold = key_map_unfold.view(key_map.shape[0], -1, key_map.shape[1], key_map_unfold.shape[-1] // key_map.shape[1]) # (N, H/sxW/s, C/m, kxk)
        query_map_unfold = query_map_unfold.view(query_map.shape[0], -1, query_map.shape[1], query_map_unfold.shape[-1] // query_map.shape[1]) # (N, H/sxW/s, C/m, kxk)
        key_map_unfold = key_map_unfold.transpose(2, 1).contiguous() # (N, C/m, H/sxW/s, kxk)
        query_map_unfold = query_map_unfold.transpose(2, 1).contiguous() # (N, C/m, H/sxW/s, kxk)
        return (key_map_unfold * query_map_unfold[:, :, :, self.k**2 // 2: self.k**2 // 2 + 1]).view(key_map_unfold.shape[0], key_map_unfold.shape[1], key_map_unfold.shape[2], self.k * self.k) # (N, C/m, H/sxW/s, kxk]


class LocalRelation(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, m=8):
        super(LocalRelation, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.m = m
        self.padding = padding
        self.key = KeyQueryMap(channels, self.m)
        self.query = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(kernel_size, self.padding, self.stride)
        self.gp = GeometricPrior(kernel_size, channels // m)
        self.unfold = nn.Unfold(kernel_size, 1, self.padding, self.stride)
        self.embedding = nn.Conv2d(channels, channels, 1)

    def forward(self, x): # (N, C, H, W)
        key = self.key(x) # (N, C/m, H, W)
        query = self.query(x) # (N, C/m, H, W)
        ac = self.ac((key, query)) # (N, C/m, H/sxW/s, kxk)
        gp = self.gp() # (1, C/m, kxk)
        weight = F.softmax(ac + gp.unsqueeze(2), dim=-1)[:, None, :, :, :] # (N, 1, C/m, H/sxW/s, kxk)
        weight = weight.view(weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3], self.k, self.k) # (N, 1, C/m, H/sxW/s, k, k)
        x_unfold = self.unfold(x).transpose(2, 1).contiguous().view(x.shape[0], -1, x.shape[1], self.k * self.k).transpose(2, 1).contiguous() # (N, C, H/sxW/s, kxk)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m, -1, self.k, self.k) # (N, m, C/m, H/sxW/s, k, k)
        output = (weight * x_unfold).view(x.shape[0], x.shape[1], -1, self.k * self.k) # (N, C, H/sxW/s, kxk)
        h = (x.shape[2] + 2 * self.padding - self.k) //  self.stride + 1
        w = (x.shape[3] + 2 * self.padding - self.k) //  self.stride + 1
        output = torch.sum(output, 3).view(x.shape[0], x.shape[1], h, w) # (N, C, H/s, W/s)
        return self.embedding(output)