"""Activation recomputation for MERIT without changing model/state shapes.

Production version of the all-region execution overlay validated in the cloud
pilots. Frozen trial helpers remain independent for report reproducibility.
"""

import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from merit_covid.models.unetrpp.epa import EPA
from merit_covid.models.merit_unetrpp.modules import (
    _check_features, master_positions, relative_offsets, sinusoidal_encoding,
)


def checkpointed_mer(self, features, master_index, z_offsets_mm=None):
    """Recompute each query block, including bias, against ALL context keys."""
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
    offsets = relative_offsets(master, depth, mode=self.position_mode,
        z_offsets_mm=z_offsets_mm, unit_mm=self.position_unit_mm)[rows[:, None], context_indices]
    pe = sinusoidal_encoding(offsets, channels, dtype=context.dtype)[:, :, None]
    q = self.query(master_features)
    k = self.key(context + self.position_scale * pe).flatten(1, 2)
    v = self.value(context).flatten(1, 2)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device=features.device),
        torch.arange(width, device=features.device), indexing="ij"), -1).reshape(-1, 2)

    def attention(query, keys, values, query_coords, all_coords, scale_raw):
        delta = (query_coords[:, None] - all_coords[None]).square().sum(-1)
        distance = (delta / float(height ** 2 + width ** 2)).repeat(1, depth - 1)
        bias = -F.softplus(scale_raw) * distance
        return F.scaled_dot_product_attention(query, keys, values, attn_mask=bias, dropout_p=0.0)

    updates = []
    for start in range(0, height * width, self.query_chunk_size):
        stop = start + self.query_chunk_size
        # Coordinates are explicit inputs: backward must not capture a later
        # loop iteration's start/stop in a closure.
        args = (q[:, start:stop], k, v, coords[start:stop], coords, self.distance_scale_raw)
        update = (checkpoint(attention, *args, use_reentrant=False, preserve_rng_state=True)
                  if self.training and torch.is_grad_enabled() else attention(*args))
        updates.append(update)
    evidence = self.output(torch.cat(updates, dim=1))
    gate = self.gate(torch.cat((master_features, evidence), -1))
    updated = master_features + self.beta * gate * evidence
    result = slices.clone()
    result[rows, master] = updated
    return result.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


def enable_activation_checkpointing(model):
    """Enable all-region recomputation once, before constructing the optimizer.

    Nonreentrant checkpointing preserves RNG for dropout. Stateful normalization
    stays outside checkpoint regions so backward does not update buffers twice.
    In evaluation/no-grad mode every wrapped region executes normally.
    """
    if getattr(model, "_activation_checkpointing_enabled", False):
        return model
    before = {name: id(value) for name, value in model.state_dict(keep_vars=True).items()}
    regions = [module for module in model.modules() if isinstance(module, EPA)]
    regions += [model.encoder1, model.decoder2]
    if model.tep is not None:
        regions.append(model.tep)
    for module in regions:
        if any(getattr(child, "track_running_stats", False) for child in module.modules()):
            raise ValueError("Checkpoint region contains mutable normalization statistics")

    def wrap(module):
        original = module.forward

        def forward(*args, **kwargs):
            if not module.training or not torch.is_grad_enabled():
                return original(*args, **kwargs)
            return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)

        module.forward = forward

    if model.mer is not None:
        model.mer.forward = types.MethodType(checkpointed_mer, model.mer)
    for module in regions:
        wrap(module)
    after = {name: id(value) for name, value in model.state_dict(keep_vars=True).items()}
    if before != after:
        raise AssertionError("Activation checkpointing changed model state identities")
    model._activation_checkpointing_enabled = True
    return model


def configure_model_execution(model, training):
    """Apply the frozen execution setting to a fresh model, leaving weights intact."""
    if training.get("activation_checkpointing", False):
        enable_activation_checkpointing(model)
    return model
