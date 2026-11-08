"""Synapse EPA from Shaker et al. (Apache-2.0); source and changes in NOTICE.

Modified: extracted from transformerblock.py and added input validation.
Attention equations, shared E/F, head layout and concatenation are unchanged.
"""

import torch
from torch import nn


class EPA(nn.Module):
    """Efficient paired channel/spatial attention on tokens [B,N,C]."""

    def __init__(self, input_size, hidden_size, proj_size, num_heads=4,
                 qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        for name, value in (("input_size", input_size), ("hidden_size", hidden_size),
                            ("proj_size", proj_size), ("num_heads", num_heads)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_heads or hidden_size % 2:
            raise ValueError("hidden_size must be even and divisible by num_heads")
        if not all(0 <= p <= 1 for p in (channel_attn_drop, spatial_attn_drop)):
            raise ValueError("Attention dropout must be in [0,1]")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        # Upstream Synapse shares this SAME layer, including its bias.
        self.E = self.F = nn.Linear(input_size, proj_size)
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        self.out_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("EPA expects a tensor [B,N,C]")
        B, N, C = x.shape
        if N != self.E.in_features or C != self.qkvv.in_features or B < 1:
            raise ValueError("EPA token/channel shape does not match its configuration")
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)
        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)
        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = self.attn_drop(attn_CA.softmax(dim=-1))
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = self.attn_drop_2(attn_SA.softmax(dim=-1))
        # Keep upstream permutation/reshape exactly, including its head layout.
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        return torch.cat((x_SA, x_CA), dim=-1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}
