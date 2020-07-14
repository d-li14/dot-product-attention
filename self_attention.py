import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(SelfAttention2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative
        self.stride = stride

        assert self.dk % self.Nh == 0, "dk should be divided by Nh."
        assert self.dv % self.Nh == 0, "dv should be divided by Nh."
        assert self.Nh > 0
        assert stride in [1, 2]

        self.conv_out = nn.Conv2d(in_channels, out_channels - dv, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.conv_kqv = nn.Conv2d(in_channels, 2 * dk + dv, kernel_size=1, stride=stride)
        self.attn_out = nn.Conv2d(dv, dv, 1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        conv_out = self.conv_out(x)
        N, _, H, W = conv_out.size()

        flat_k, flat_q, flat_v, k, q, v = self.compute_flat_kqv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            rel_logits_h, rel_logits_w = self.relative_logits(q)
            logits += rel_logits_h
            logits += rel_logits_w
        weights = F.softmax(logits, dim=-1)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (N, self.Nh, self.dv // self.Nh, H, W))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_kqv(self, x, dk, dv, Nh):
        dkh = dk // Nh
        dvh = dv // Nh
        kqv = self.conv_kqv(x)
        N, _, H, W = kqv.size()
        k, q, v = torch.split(kqv, [dk, dk, dv], dim=1)
        q *= dkh ** -0.5
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        flat_q = torch.reshape(q, (N, Nh, dkh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dkh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dvh, H * W))
        return flat_k, flat_q, flat_v, k, q, v

    def split_heads_2d(self, x, Nh):
        N, C, H, W = x.size()
        ret_shape = (N, Nh, C // Nh, H, W)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        N, Nh, dv, H, W = x.size()
        ret_shape = (N, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        N, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x