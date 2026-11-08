"""Synapse hierarchical encoder, Shaker et al., Apache-2.0; see NOTICE.

Modified: split from model_components.py, tuple defaults, local imports,
torch trunc_normal_ in place of timm, and explicit einops dependency.
"""

import einops
from monai.networks.layers.utils import get_norm_layer
from torch import nn
from torch.nn.init import trunc_normal_

from .blocks import get_conv_layer
from .transformer import TransformerBlock


class UnetrPPEncoder(nn.Module):
    """Patch stem (2,4,4), then three (2,2,2) downsampling stages."""

    def __init__(self, input_size=(32768, 4096, 512, 64),
                 dims=(32, 64, 128, 256), proj_size=(64, 64, 64, 32),
                 depths=(3, 3, 3, 3), num_heads=4, spatial_dims=3,
                 in_channels=1, dropout=0.0, transformer_dropout_rate=0.15):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("UNETR++ encoder requires spatial_dims=3")
        if not all(len(v) == 4 for v in (input_size, dims, proj_size, depths)):
            raise ValueError("Encoder needs four input sizes/dims/projection sizes/depths")
        if any(type(d) is not int or d < 1 for d in depths):
            raise ValueError("Each encoder depth must be a positive integer")
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            get_conv_layer(3, in_channels, dims[0], kernel_size=(2, 4, 4),
                           stride=(2, 4, 4), dropout=dropout, conv_only=True),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                get_conv_layer(3, dims[i], dims[i + 1], kernel_size=2,
                               stride=2, dropout=dropout, conv_only=True),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            ))
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                TransformerBlock(input_size[i], dims[i], proj_size[i], num_heads,
                                 dropout_rate=transformer_dropout_rate, pos_embed=True)
                for _ in range(depths[i])
            ]))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Upstream initializes Conv2d/Linear, NOT Conv3d, here.
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            if i == 3:
                x = einops.rearrange(x, "b c d h w -> b (d h w) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        return self.forward_features(x)
