"""Z-preserving UNETR++ with the paper's three MERIT insertion points."""

import math
from typing import NamedTuple

import torch

from ..unetrpp.model import UNETRPP
from .modules import (
    MasterEvidenceRouting, PresenceCommitmentGate, TrajectoryEvidencePrototype,
    master_positions,
    relative_offsets,
)


class MERITOutput(NamedTuple):
    """The paper's four-tuple. Disabled auxiliary modules return None."""

    logits: torch.Tensor
    presence: torch.Tensor | None
    salience: torch.Tensor | None
    raw_logits: torch.Tensor


class MERITUNETRPP(UNETRPP):
    """Binary slab segmentation [B,2,Z,H,W], conditioned on per-sample masters.

    Default grid (11,128,128) is from Sec. 3.1/Fig. 1. Every encoder/decoder
    stage preserves Z. The original UNETRPP default remains a separate baseline.
    No interpolation, crop, normalization or ground truth is used in forward.
    """

    def __init__(self, in_channels=1, out_channels=2, *, img_size=(11, 128, 128),
                 feature_size=16, depths=(3, 3, 3, 3), num_heads=4,
                 norm_name="instance", use_tep=True, use_mer=True, use_pcg=True,
                 salience_hidden=64, position_dim=64, position_hidden=16,
                 mer_attention_dim=64, gate_hidden=16, presence_hidden=8,
                 query_chunk_size=128, pcg_ramp=1.0, position_mode="index", position_unit_mm=1.0):
        for name, value in (("use_tep", use_tep), ("use_mer", use_mer), ("use_pcg", use_pcg)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if out_channels != 2:
            raise ValueError("MERIT uses exactly two channels: background and foreground")
        if position_mode not in ("index", "physical_mm") or not math.isfinite(position_unit_mm) or position_unit_mm <= 0:
            raise ValueError("Invalid position mode/unit")
        self.position_mode, self.position_unit_mm = position_mode, position_unit_mm
        super().__init__(in_channels, out_channels, img_size=img_size, feature_size=feature_size,
                         num_heads=num_heads, norm_name=norm_name, depths=depths,
                         do_ds=False, preserve_z=True)
        if use_mer and self.img_size[0] < 2:
            raise ValueError("MER needs at least two input slices")
        self.tep = TrajectoryEvidencePrototype(self.hidden_size, salience_hidden=salience_hidden,
            position_dim=position_dim, position_hidden=position_hidden,
            position_mode=position_mode, position_unit_mm=position_unit_mm) if use_tep else None
        self.mer = MasterEvidenceRouting(self.dims[0], attention_dim=mer_attention_dim,
            gate_hidden=gate_hidden, query_chunk_size=query_chunk_size,
            position_mode=position_mode, position_unit_mm=position_unit_mm) if use_mer else None
        self.pcg = PresenceCommitmentGate(feature_size, hidden=presence_hidden) if use_pcg else None
        self.register_buffer("pcg_ramp", torch.tensor(1.0))
        self.set_pcg_ramp(pcg_ramp)

    def set_pcg_ramp(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("pcg_ramp must be in [0,1]")
        self.pcg_ramp.fill_(value)

    def set_epoch(self, epoch, *, warmup_epochs=30):
        """Explicit 0-based warmup: epoch 0 -> 0, epoch 30 onward -> 1.

        Forward never advances the schedule. The ramp is saved in state_dict;
        a newly constructed model defaults to full correction for inference.
        """
        if (not math.isfinite(epoch) or epoch < 0 or not math.isfinite(warmup_epochs)
                or warmup_epochs <= 0):
            raise ValueError("epoch must be nonnegative and warmup_epochs positive")
        self.set_pcg_ramp(min(float(epoch) / warmup_epochs, 1.0))

    def forward(self, x, master_index, z_offsets_mm=None):
        self.validate_input(x)
        master = master_positions(master_index, x.shape[0], x.shape[2], x.device)
        relative_offsets(master, x.shape[2], mode=self.position_mode,
                         z_offsets_mm=z_offsets_mm, unit_mm=self.position_unit_mm)
        _, hidden = self.unetr_pp_encoder(x)
        enc1, enc2, enc3, enc4 = hidden
        bottleneck = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        salience = None
        if self.tep is not None:
            bottleneck, salience = self.tep(bottleneck, master, z_offsets_mm)
        dec3 = self.decoder5(bottleneck, enc3)
        dec2 = self.decoder4(dec3, enc2)
        mid = self.decoder3(dec2, enc1)
        if self.mer is not None:
            mid = self.mer(mid, master, z_offsets_mm)
        full = self.decoder2(mid, self.encoder1(x))
        raw = self.out1(full)
        logits, presence = (raw, None) if self.pcg is None else self.pcg(full, raw, self.pcg_ramp)
        return MERITOutput(logits, presence, salience, raw)
