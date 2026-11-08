"""Synapse upsampling block, Shaker et al., Apache-2.0; see NOTICE.

Modified: local imports, formatting and an explicit skip-shape check.
Transposed convolution, ADDITIVE skip and transformer/conv paths are unchanged.
"""

from torch import nn

from .blocks import get_conv_layer, UnetResBlock
from .transformer import TransformerBlock


class UnetrUpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size,
                 upsample_kernel_size, norm_name, proj_size=64, num_heads=4,
                 out_size=0, depth=3, conv_decoder=False):
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("UNETR++ decoder requires spatial_dims=3")
        if type(depth) is not int or depth < 1:
            raise ValueError("Decoder depth must be a positive integer")
        self.transp_conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size, conv_only=True, is_transposed=True,
        )
        self.decoder_block = nn.ModuleList()
        if conv_decoder:
            self.decoder_block.append(UnetResBlock(
                spatial_dims, out_channels, out_channels, kernel_size, 1, norm_name,
            ))
        else:
            self.decoder_block.append(nn.Sequential(*[
                TransformerBlock(out_size, out_channels, proj_size, num_heads,
                                 dropout_rate=0.15, pos_embed=True)
                for _ in range(depth)
            ]))

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        if out.shape != skip.shape:
            raise ValueError(f"Decoder skip shape mismatch: {tuple(out.shape)} vs {tuple(skip.shape)}")
        return self.decoder_block[0](out + skip)
