"""TEP, MER and PCG from the supplied MERIT paper, Eqs. (1)-(12), Fig. 1.

Paper-derived reimplementation, not the authors' source. See docs/merit_unetrpp.md
for dimensions, parameter-count checks and explicitly unspecified choices.
"""

import math
import operator

import torch
from torch import nn
from torch.nn import functional as F


def master_positions(indices, batch_size, depth, device):
    """Validate slab-local positions; never interpret volume slice IDs as positions."""
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise ValueError("master_index must be a vector of integer slab positions")
        indices = indices.tolist()
    if not isinstance(indices, (list, tuple)) or len(indices) != batch_size:
        raise ValueError("One integer master_index is required per batch element")
    positions = []
    for value in indices:
        try:
            if isinstance(value, bool):
                raise TypeError
            index = operator.index(value)
        except TypeError:
            raise ValueError("master_index contains a non-integer slab position") from None
        if not 0 <= index < depth:
            raise ValueError("master_index is outside the slab")
        positions.append(index)
    return torch.tensor(positions, dtype=torch.long, device=device)


def sinusoidal_encoding(offsets, channels, *, dtype=torch.float32):
    """Fixed PE(z-m), interleaved sin/cos, frequencies 10000**(-2i/C)."""
    if type(channels) is not int or channels < 1:
        raise ValueError("PE channels must be positive")
    # Compute safely under mixed precision, then match the feature dtype.
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    frequencies = torch.exp(torch.arange(0, channels, 2, device=offsets.device,
                                         dtype=work_dtype) * (-math.log(10000.0) / channels))
    angles = offsets.to(work_dtype)[..., None] * frequencies
    encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
    return encoding[..., :channels].to(dtype)


def _check_features(features, channels):
    if (not isinstance(features, torch.Tensor) or features.ndim != 5
            or features.shape[1] != channels or min(features.shape) < 1
            or not features.is_floating_point()):
        raise ValueError(f"Expected floating features [B,{channels},Z,H,W]")


def _positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def relative_offsets(master, depth, *, mode="index", z_offsets_mm=None, unit_mm=1.0):
    """Physical PE arguments are actual mm / explicit unit_mm, not target mm."""
    if mode == "index":
        return torch.arange(depth, device=master.device)[None] - master[:, None]
    if mode != "physical_mm" or not math.isfinite(unit_mm) or unit_mm <= 0:
        raise ValueError("Invalid position mode or physical unit scale")
    if z_offsets_mm is None:
        raise ValueError("physical_mm requires actual z_offsets_mm; no index fallback")
    values = torch.as_tensor(z_offsets_mm, device=master.device, dtype=torch.float32)
    if values.shape != (len(master), depth) or not torch.isfinite(values).all():
        raise ValueError("z_offsets_mm must be finite [B,Z]")
    if not torch.allclose(values[torch.arange(len(master), device=master.device), master],
                          torch.zeros(len(master), device=master.device), atol=1e-4, rtol=0):
        raise ValueError("z_offsets_mm at master must equal zero")
    return values / unit_mm


class TrajectoryEvidencePrototype(nn.Module):
    """Content salience -> master-relative pooling -> per-slice prototype attention."""

    def __init__(self, channels=256, *, salience_hidden=64, position_dim=64,
                 position_hidden=16, position_mode="index", position_unit_mm=1.0):
        super().__init__()
        for name, value in (("channels", channels), ("salience_hidden", salience_hidden),
                            ("position_dim", position_dim), ("position_hidden", position_hidden)):
            _positive_integer(name, value)
        if position_mode not in ("index", "physical_mm") or not math.isfinite(position_unit_mm) or position_unit_mm <= 0:
            raise ValueError("Invalid position mode/unit")
        self.position_mode, self.position_unit_mm = position_mode, position_unit_mm
        self.channels = channels
        self.position_dim = position_dim
        self.salience_head = nn.Sequential(nn.Linear(channels, salience_hidden), nn.GELU(),
                                           nn.Linear(salience_hidden, 1))
        self.position_head = nn.Sequential(nn.Linear(1 + position_dim, position_hidden), nn.GELU(),
                                           nn.Linear(position_hidden, 1))
        self.tau0 = nn.Parameter(torch.zeros(()))
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)

    @property
    def temperature(self):
        return F.softplus(self.tau0) + 0.1

    def forward(self, features, master_index, z_offsets_mm=None):
        _check_features(features, self.channels)
        batch, channels, depth, height, width = features.shape
        master = master_positions(master_index, batch, depth, features.device)
        slices = features.permute(0, 2, 3, 4, 1).reshape(batch, depth, height * width, channels)
        salience = self.salience_head(slices.mean(2)).squeeze(-1).float().sigmoid()
        offsets = relative_offsets(master, depth, mode=self.position_mode,
                                   z_offsets_mm=z_offsets_mm, unit_mm=self.position_unit_mm)
        position = sinusoidal_encoding(offsets, self.position_dim, dtype=slices.dtype)
        scores = self.position_head(torch.cat((salience[..., None].to(slices.dtype), position), -1)).squeeze(-1)
        weights = (scores.float() / self.temperature.float()).softmax(dim=1).to(slices.dtype)
        prototype = (weights[:, :, None, None] * slices).sum(dim=1)
        q = self.query(slices)
        k, v = self.key(prototype), self.value(prototype)
        # Independent N x N spatial attention for each slice, not (Z*N)^2.
        update = F.scaled_dot_product_attention(q, k[:, None], v[:, None], dropout_p=0.0)
        refined = slices + self.output(update)
        return refined.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3).contiguous(), salience


class MasterEvidenceRouting(nn.Module):
    """Eqs. (5)-(9): context-only keys/values; only the designated master changes.

    Query chunks bound individual score allocations, not total training memory:
    autograd may retain histories across chunks. Each chunk attends over ALL
    context tokens with one joint softmax; this is not local/windowed attention.
    """

    def __init__(self, channels=32, *, attention_dim=64, gate_hidden=16, query_chunk_size=128,
                 position_mode="index", position_unit_mm=1.0):
        super().__init__()
        for name, value in (("channels", channels), ("attention_dim", attention_dim),
                            ("gate_hidden", gate_hidden), ("query_chunk_size", query_chunk_size)):
            _positive_integer(name, value)
        self.channels = channels
        if position_mode not in ("index", "physical_mm") or not math.isfinite(position_unit_mm) or position_unit_mm <= 0:
            raise ValueError("Invalid position mode/unit")
        self.position_mode, self.position_unit_mm = position_mode, position_unit_mm
        self.attention_dim = attention_dim
        self.query_chunk_size = query_chunk_size
        self.query = nn.Linear(channels, attention_dim, bias=False)
        self.key = nn.Linear(channels, attention_dim, bias=False)
        self.value = nn.Linear(channels, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, channels, bias=False)
        self.gate = nn.Sequential(nn.Linear(channels * 2, gate_hidden), nn.GELU(),
                                  nn.Linear(gate_hidden, 1), nn.Sigmoid())
        # Softplus constrains lambda_pe and gamma_xy; beta is an unconstrained scale.
        initial = math.log(math.expm1(1.0))
        self.position_scale_raw = nn.Parameter(torch.tensor(initial))
        self.distance_scale_raw = nn.Parameter(torch.tensor(initial))
        self.beta = nn.Parameter(torch.tensor(0.1))

    @property
    def position_scale(self):
        return F.softplus(self.position_scale_raw)

    @property
    def distance_scale(self):
        return F.softplus(self.distance_scale_raw)

    def forward(self, features, master_index, z_offsets_mm=None):
        _check_features(features, self.channels)
        batch, channels, depth, height, width = features.shape
        if depth < 2:
            raise ValueError("MER needs at least one context slice")
        master = master_positions(master_index, batch, depth, features.device)
        rows = torch.arange(batch, device=features.device)
        positions = torch.arange(depth, device=features.device).expand(batch, -1)
        context_indices = positions[positions != master[:, None]].reshape(batch, depth - 1)
        slices = features.permute(0, 2, 3, 4, 1).reshape(batch, depth, height * width, channels)
        master_features = slices[rows, master]
        context = slices[rows[:, None], context_indices]
        all_offsets = relative_offsets(master, depth, mode=self.position_mode,
                                       z_offsets_mm=z_offsets_mm, unit_mm=self.position_unit_mm)
        offsets = all_offsets[rows[:, None], context_indices]
        pe = sinusoidal_encoding(offsets, channels, dtype=context.dtype)[:, :, None]
        q = self.query(master_features)
        k = self.key(context + self.position_scale * pe).flatten(1, 2)
        v = self.value(context).flatten(1, 2)  # No PE in the value branch.
        coords = torch.stack(torch.meshgrid(torch.arange(height, device=features.device),
                                            torch.arange(width, device=features.device), indexing="ij"), -1).reshape(-1, 2)
        updates = []
        for start in range(0, height * width, self.query_chunk_size):
            stop = start + self.query_chunk_size
            delta = (coords[start:stop, None] - coords[None]).square().sum(-1)
            distance = (delta / float(height ** 2 + width ** 2)).repeat(1, depth - 1)
            bias = -self.distance_scale * distance
            evidence = F.scaled_dot_product_attention(q[:, start:stop], k, v,
                                                      attn_mask=bias, dropout_p=0.0)
            updates.append(evidence)
        evidence = self.output(torch.cat(updates, dim=1))
        gate = self.gate(torch.cat((master_features, evidence), dim=-1))
        updated = master_features + self.beta * gate * evidence
        result = slices.clone()
        result[rows, master] = updated
        return result.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


class PresenceCommitmentGate(nn.Module):
    """Eqs. (10)-(12): preserve background, add slice log-odds to foreground."""

    def __init__(self, channels=16, *, hidden=8, epsilon=1e-6):
        super().__init__()
        _positive_integer("channels", channels)
        _positive_integer("hidden", hidden)
        if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
            raise ValueError("epsilon must be in (0,0.5)")
        self.channels, self.epsilon = channels, epsilon
        self.presence_head = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.presence_head[-1].bias, math.log(0.7 / 0.3))

    def forward(self, features, raw_logits, ramp=1.0):
        _check_features(features, self.channels)
        if raw_logits.shape != (features.shape[0], 2, *features.shape[2:]):
            raise ValueError("PCG needs two-channel logits aligned with its features")
        strength = float(ramp)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("PCG ramp must be in [0,1]")
        descriptors = features.mean(dim=(-2, -1)).transpose(1, 2)
        presence = self.presence_head(descriptors).squeeze(-1).float().sigmoid()
        safe = presence.clamp(self.epsilon, 1 - self.epsilon)
        correction = strength * (safe.log() - torch.log1p(-safe))
        foreground = raw_logits[:, 1:2] + correction[:, None, :, None, None]
        return torch.cat((raw_logits[:, :1], foreground), dim=1), presence
