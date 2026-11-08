"""Synapse transformer block, Shaker et al., Apache-2.0; see NOTICE.

Modified: local imports, explicit D/H/W names and a shape check. The residual
path, positional embeddings, EPA, gamma and convolution path are unchanged.
"""

import torch
from torch import nn

from .blocks import UnetResBlock
from .epa import EPA


class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads,
                 dropout_rate=0.0, pos_embed=False):
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be in [0,1]")
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size))
        self.epa_block = EPA(input_size, hidden_size, proj_size, num_heads,
                             channel_attn_drop=dropout_rate,
                             spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, 3, 1, "batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("TransformerBlock expects [B,C,D,H,W]")
        B, C, D, H, W = x.shape
        x = x.reshape(B, C, D * H * W).permute(0, 2, 1)
        if self.pos_embed is not None:
            if x.shape[1:] != self.pos_embed.shape[1:]:
                raise ValueError("Transformer token/channel shape differs from positional embedding")
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))
        attn_skip = attn.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        attn = self.conv51(attn_skip)
        return attn_skip + self.conv8(attn)
