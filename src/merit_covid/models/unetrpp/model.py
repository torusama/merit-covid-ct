"""UNETR++ Synapse architecture, Shaker et al., Apache-2.0; see NOTICE.

Modified: nn.Module replaces the nnU-Net training/inference superclass;
sizes are derived from an explicit img_size; validation and API naming added.
Default topology/parameters and forward operations follow upstream Synapse.
"""

from math import prod
from collections.abc import Sequence

import torch
from torch import nn

from .blocks import UnetOutBlock, UnetResBlock
from .decoder import UnetrUpBlock
from .encoder import UnetrPPEncoder


def _positive_tuple(name, values, length):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain {length} positive integers")
    values = tuple(values)
    if len(values) != length or any(type(v) is not int or v < 1 for v in values):
        raise ValueError(f"{name} must contain {length} positive integers")
    return values


class UNETRPP(nn.Module):
    """3D raw logits; do_ds=True returns [full, patch-grid, next-coarser].

    img_size is a FIXED token grid, not a resize instruction. Default is the
    upstream Synapse (64,128,128). Other explicitly configured grids preserve
    the operations but are adaptations, not published experiment settings.
    """

    def __init__(self, in_channels=1, out_channels=1, *, img_size=(64, 128, 128),
                 feature_size=16, hidden_size=None, num_heads=4,
                 norm_name="instance", depths=(3, 3, 3, 3), dims=None,
                 do_ds=True, preserve_z=False):
        super().__init__()
        self.img_size = _positive_tuple("img_size", img_size, 3)
        self.depths = _positive_tuple("depths", depths, 4)
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels),
                            ("feature_size", feature_size), ("num_heads", num_heads)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(do_ds) is not bool:
            raise ValueError("do_ds must be a boolean")
        if type(preserve_z) is not bool:
            raise ValueError("preserve_z must be a boolean")
        self.preserve_z = preserve_z
        self.patch_size = (1, 4, 4) if preserve_z else (2, 4, 4)
        divisors = (1, 32, 32) if preserve_z else (16, 32, 32)
        if any(s % d for s, d in zip(self.img_size, divisors)):
            raise ValueError(f"img_size (D,H,W) must be divisible by {divisors}; no automatic resizing/padding")
        self.feat_size = tuple(s // d for s, d in zip(self.img_size, divisors))
        if prod(self.feat_size) < 2:
            raise ValueError("Deepest grid needs at least two voxels for upstream normalization")
        expected_dims = tuple(feature_size * m for m in (2, 4, 8, 16))
        self.dims = expected_dims if dims is None else _positive_tuple("dims", dims, 4)
        if self.dims != expected_dims:
            raise ValueError("dims must equal feature_size * (2,4,8,16) to match decoder skips")
        if any(d % num_heads for d in self.dims):
            raise ValueError("Encoder dims must be divisible by num_heads")
        # Upstream decoder uses four heads independently of encoder num_heads.
        if any(d % 4 for d in self.dims[:3]) or self.dims[0] % in_channels:
            raise ValueError("Channels must support decoder heads=4 and stem GroupNorm groups=in_channels")
        if hidden_size is not None and (type(hidden_size) is not int or hidden_size != self.dims[-1]):
            raise ValueError("hidden_size must equal the last encoder dimension")
        self.hidden_size = self.dims[-1]
        self.in_channels = in_channels
        self.num_classes = out_channels
        self.do_ds = do_ds
        self.conv_op = nn.Conv3d
        grids = [tuple(s // (p * (1 if preserve_z and axis == 0 else 2 ** i))
                       for axis, (s, p) in enumerate(zip(self.img_size, self.patch_size)))
                 for i in range(4)]
        self.stage_shapes = tuple(grids)
        tokens = tuple(prod(g) for g in grids)
        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=tokens, dims=self.dims, depths=self.depths,
            num_heads=num_heads, in_channels=in_channels,
            preserve_z=preserve_z,
        )
        self.encoder1 = UnetResBlock(3, in_channels, feature_size, 3, 1, norm_name)
        up_stride = (1, 2, 2) if preserve_z else 2
        self.decoder5 = UnetrUpBlock(3, self.dims[3], self.dims[2], 3, up_stride, norm_name, out_size=tokens[2])
        self.decoder4 = UnetrUpBlock(3, self.dims[2], self.dims[1], 3, up_stride, norm_name, out_size=tokens[1])
        self.decoder3 = UnetrUpBlock(3, self.dims[1], self.dims[0], 3, up_stride, norm_name, out_size=tokens[0])
        self.decoder2 = UnetrUpBlock(
            3, self.dims[0], feature_size, 3, self.patch_size, norm_name,
            out_size=prod(self.img_size), conv_decoder=True,
        )
        self.out1 = UnetOutBlock(3, feature_size, out_channels)
        if do_ds:
            self.out2 = UnetOutBlock(3, self.dims[0], out_channels)
            self.out3 = UnetOutBlock(3, self.dims[1], out_channels)

    @staticmethod
    def proj_feat(x, hidden_size, feat_size):
        x = x.reshape(x.shape[0], *feat_size, hidden_size)
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def validate_input(self, x):
        """Reject incompatible data before running convolutions or allocating attention."""
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("UNETRPP expects a torch tensor [B,C,D,H,W]")
        if x.shape[0] < 1 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected nonempty batch with {self.in_channels} input channels")
        if tuple(x.shape[2:]) != self.img_size:
            raise ValueError(
                f"Expected configured img_size={self.img_size}, got {tuple(x.shape[2:])}. "
                "A different slab shape is not directly supported; "
                "input preparation requires a separate training protocol. No automatic resize/pad."
            )
        if not x.is_floating_point():
            raise ValueError("UNETRPP input must have a floating dtype")

    def forward(self, x):
        self.validate_input(x)
        _, hidden_states = self.unetr_pp_encoder(x)
        conv_block = self.encoder1(x)
        enc1, enc2, enc3, enc4 = hidden_states
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, conv_block)
        if self.do_ds:
            return [self.out1(out), self.out2(dec1), self.out3(dec2)]
        return self.out1(out)
