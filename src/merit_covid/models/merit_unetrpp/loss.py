"""Sec. 3.5/Table 1 loss with explicit choices where the PDF is underspecified.

Requires actual 3D slab/patch GT; a single master mask is insufficient. Padding
is excluded using valid_region. See docs/merit_unetrpp.md for the exact policy.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .modules import master_positions, relative_offsets


def probability_bce(probability, target):
    """Keep the existing probability BCE, explicitly outside CUDA autocast.

    The paper's four-tuple supplies probabilities, not auxiliary raw logits.
    Float casts alone do not disable CUDA's autocast ban on BCELoss.
    """
    with torch.autocast(device_type=probability.device.type, enabled=False):
        return F.binary_cross_entropy(probability.float(), target.float(), reduction="none")


class MERITLoss(nn.Module):
    def __init__(self, *, gaussian_sigma=2.0, presence_weight=0.2,
                 alignment_weight=0.05, salience_weight=0.03,
                 presence_positive_weight=1.0, smooth=1e-5, weighting_mode="index",
                 sigma_slices=None, sigma_mm=None, alignment_pool="max", top_k_fraction=0.05):
        super().__init__()
        for name, value in (("gaussian_sigma", gaussian_sigma),
                            ("presence_positive_weight", presence_positive_weight), ("smooth", smooth)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("presence_weight", presence_weight), ("alignment_weight", alignment_weight),
                            ("salience_weight", salience_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if weighting_mode not in ("index", "physical_mm"):
            raise ValueError("Invalid weighting_mode")
        sigma_slices = gaussian_sigma if sigma_slices is None else sigma_slices
        if not math.isfinite(sigma_slices) or sigma_slices <= 0:
            raise ValueError("sigma_slices must be positive")
        if weighting_mode == "physical_mm" and (sigma_mm is None or not math.isfinite(sigma_mm) or sigma_mm <= 0):
            raise ValueError("Physical weighting requires explicit positive sigma_mm")
        if weighting_mode == "index" and sigma_mm is not None:
            raise ValueError("sigma_mm is only used by physical_mm weighting")
        if alignment_pool not in ("max", "top_k_mean") or not 0 < top_k_fraction <= 1:
            raise ValueError("Invalid alignment pooling configuration")
        self.weighting_mode, self.sigma_mm = weighting_mode, sigma_mm
        self.gaussian_sigma = sigma_slices  # Legacy API alias, units are slices.
        self.alignment_pool, self.top_k_fraction = alignment_pool, top_k_fraction
        self.presence_weight = presence_weight
        self.alignment_weight = alignment_weight
        self.salience_weight = salience_weight
        self.presence_positive_weight = presence_positive_weight
        self.smooth = smooth

    def forward(self, output, target, master_index, *, valid_region=None, z_offsets_mm=None):
        logits, presence, salience, raw = output
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 5 or logits.shape[1] != 2
                or min(logits.shape) < 1 or raw.shape != logits.shape):
            raise ValueError("Expected MERIT four-tuple with logits [B,2,Z,H,W]")
        batch, _, depth, height, width = logits.shape
        master = master_positions(master_index, batch, depth, logits.device)
        shape = (batch, depth, height, width)
        if not isinstance(target, torch.Tensor):
            raise ValueError("target must be a binary torch tensor with actual full-slab GT")
        if target.shape == (batch, 1, depth, height, width):
            target = target[:, 0]
        if target.shape != shape:
            raise ValueError("Expected actual full-slab GT [B,1,Z,H,W] or [B,Z,H,W], not a repeated master mask")
        if valid_region is None:
            valid = torch.ones(shape, device=logits.device, dtype=torch.bool)
        else:
            if valid_region.shape == (batch, 1, depth, height, width):
                valid_region = valid_region[:, 0]
            if valid_region.shape != shape or valid_region.dtype != torch.bool:
                raise ValueError("valid_region must be bool with the full-slab target shape")
            valid = valid_region.to(logits.device)
        target = target.to(logits.device)
        if not valid.any() or not torch.all((target[valid] == 0) | (target[valid] == 1)):
            raise ValueError("Need at least one valid voxel and binary 0/1 labels on valid voxels")
        target = torch.where(valid, target, 0).long()
        truth = target.float()
        pixels = valid.sum(dim=(-2, -1))
        active = pixels > 0
        if not active[torch.arange(batch, device=logits.device), master].all():
            raise ValueError("Every master must contain valid real voxels")
        offsets = relative_offsets(master, depth, mode=self.weighting_mode, z_offsets_mm=z_offsets_mm)
        sigma = self.sigma_mm if self.weighting_mode == "physical_mm" else self.gaussian_sigma
        # Float64 avoids intermediate overflow/0*inf for extreme positive sigma.
        weights = torch.exp(-0.5 * (offsets.double() / sigma).square()).float() * active
        normalizer = weights.sum().clamp_min(torch.finfo(torch.float32).tiny)
        probabilities = logits.float().softmax(1)[:, 1] * valid
        intersection = (probabilities * truth).sum(dim=(-2, -1))
        dice = 1 - (2 * intersection + self.smooth) / (
            probabilities.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + self.smooth)
        ce = F.cross_entropy(logits.float(), target, reduction="none")
        ce = (ce * valid).sum(dim=(-2, -1)) / pixels.clamp_min(1)
        segmentation = ((dice + ce) * weights).sum() / normalizer
        slice_truth = ((target == 1) & valid).any(dim=(-2, -1)).float()
        # A full-resolution FP16 sum may overflow even for finite logits.
        # Disabled auxiliary losses need a scalar zero, not a tensor reduction.
        zero = logits.new_zeros((), dtype=torch.float32)
        presence_loss, alignment, salience_loss = zero, zero, zero
        if presence is not None:
            if presence.shape != (batch, depth):
                raise ValueError("presence must have shape [B,Z]")
            presence = presence.float()
            bce = probability_bce(presence, slice_truth)
            positive_weight = torch.where(slice_truth.bool(), self.presence_positive_weight, 1.0)
            presence_loss = (bce * positive_weight * weights).sum() / normalizer
            # Explicit implementation choice: max RAW foreground probability
            # versus presence, with MSE. Never use PCG-corrected probabilities.
            raw_probability = raw.float().softmax(1)[:, 1].masked_fill(~valid, 0)
            if self.alignment_pool == "max":
                raw_presence = raw_probability.amax(dim=(-2, -1))
            else:
                pooled = []
                for b in range(batch):
                    planes = []
                    for z in range(depth):
                        values = raw_probability[b, z][valid[b, z]]
                        k = max(1, math.ceil(values.numel() * self.top_k_fraction))
                        planes.append(values.topk(k).values.mean() if values.numel() else raw_probability[b, z].sum() * 0)
                    pooled.append(torch.stack(planes))
                raw_presence = torch.stack(pooled)
            alignment = ((raw_presence - presence).square() * active).sum() / active.sum()
        if salience is not None:
            if salience.shape != (batch, depth):
                raise ValueError("salience must have shape [B,Z]")
            bce = probability_bce(salience, slice_truth)
            salience_loss = (bce * active).sum() / active.sum()
        total = (segmentation + self.presence_weight * presence_loss
                 + self.alignment_weight * alignment + self.salience_weight * salience_loss)
        return {"total": total, "segmentation": segmentation, "presence": presence_loss,
                "alignment": alignment, "salience": salience_loss}
