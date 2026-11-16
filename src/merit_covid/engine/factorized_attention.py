"""Diagnostic MER variant that factors squared XY bias into query/key.

This preserves the full-context softmax in exact arithmetic. CUDA reduced
precision can introduce different rounding; validate outputs and gradients.
"""
import math
import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from merit_covid.models.merit_unetrpp.modules import (
    _attention_validity, _check_features, master_positions, relative_offsets, sinusoidal_encoding,
)


def factorized_mer(self, features, master_index, z_offsets_mm=None, attention_valid=None):
    _check_features(features, self.channels)
    batch, channels, depth, height, width = features.shape
    if depth < 2:
        raise ValueError("MER needs at least one context slice")
    master = master_positions(master_index, batch, depth, features.device)
    rows = torch.arange(batch, device=features.device)
    positions = torch.arange(depth, device=features.device).expand(batch, -1)
    context_indices = positions[positions != master[:, None]].reshape(batch, depth - 1)
    valid = _attention_validity(attention_valid, master, depth)
    context_valid = None if valid is None else valid[rows[:, None], context_indices]
    slices = features.permute(0, 2, 3, 4, 1).reshape(batch, depth, height * width, channels)
    master_features = slices[rows, master]
    context = slices[rows[:, None], context_indices]
    offsets = relative_offsets(master, depth, mode=self.position_mode,
        z_offsets_mm=z_offsets_mm, unit_mm=self.position_unit_mm)[rows[:, None], context_indices]
    pe = sinusoidal_encoding(offsets, channels, dtype=context.dtype)[:, :, None]
    query = self.query(master_features)
    keys = self.key(context + self.position_scale * pe).flatten(1, 2)
    values = self.value(context).flatten(1, 2)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device=features.device),
        torch.arange(width, device=features.device), indexing="ij"), -1).reshape(-1, 2)
    # ||x-y||²/(H²+W²): remove query-only term because softmax is invariant to it.
    coords = coords.to(torch.float32) / math.sqrt(height ** 2 + width ** 2)
    key_positions = coords.repeat(depth - 1, 1)
    extra_keys = torch.cat((key_positions, key_positions.square().sum(-1, keepdim=True)), dim=-1)
    padded_dim = ((self.attention_dim + 3 + 7) // 8) * 8
    extra_padding = padded_dim - self.attention_dim - 3
    keys = torch.cat((keys, F.pad(extra_keys.to(keys.dtype), (0, extra_padding))[None].expand(batch, -1, -1)), dim=-1)
    mask = has_context = None
    if context_valid is not None:
        token_valid = context_valid.repeat_interleave(height * width, dim=1)
        has_context = token_valid.any(dim=1)
        safe_valid = token_valid.clone()
        safe_valid[~has_context, 0] = True
        mask = safe_valid[:, None, None, :]

    def attention(block, all_keys, all_values, block_coords, scale_raw):
        scale = F.softplus(scale_raw).float()
        position = torch.cat((2 * scale * block_coords, -scale * torch.ones_like(block_coords[:, :1])), dim=-1)
        # Keep the original 1/sqrt(d) attention scale explicitly: rescaling
        # learned query channels under FP16 introduces avoidable rounding.
        position = F.pad(position * math.sqrt(self.attention_dim), (0, extra_padding)).to(block.dtype)
        padded_query = torch.cat((block, position[None].expand(batch, -1, -1)), dim=-1)
        result = F.scaled_dot_product_attention(padded_query[:, None], all_keys[:, None], all_values[:, None],
            attn_mask=mask, dropout_p=0.0, scale=1 / math.sqrt(self.attention_dim)).squeeze(1)
        return result if has_context is None else result * has_context[:, None, None]

    updates = []
    for start in range(0, height * width, self.query_chunk_size):
        stop = start + self.query_chunk_size
        args = (query[:, start:stop], keys, values, coords[start:stop], self.distance_scale_raw)
        result = (checkpoint(attention, *args, use_reentrant=False, preserve_rng_state=True)
                  if self.training and torch.is_grad_enabled() else attention(*args))
        updates.append(result)
    evidence = self.output(torch.cat(updates, dim=1))
    gate = self.gate(torch.cat((master_features, evidence), dim=-1))
    updated = master_features + self.beta * gate * evidence
    result = slices.clone()
    result[rows, master] = updated
    return result.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


def enable_factorized_mer(model):
    if model.mer is None:
        raise ValueError("Model needs MER")
    model.mer.forward = types.MethodType(factorized_mer, model.mer)
    return model
