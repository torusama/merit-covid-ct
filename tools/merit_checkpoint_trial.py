"""Experimental activation checkpointing; production model source is unchanged.

Run in a fresh process. The cloud notebook embeds this file so unpublished
runtime experiments can be tested against its pinned public model source.
"""

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss, MasterEvidenceRouting
from merit_covid.models.unetrpp.epa import EPA
from merit_covid.models.merit_unetrpp.modules import (
    _check_features, master_positions, relative_offsets, sinusoidal_encoding,
)


def checkpointed_mer(self, features, master_index, z_offsets_mm=None):
    """Original MER equations; recompute distance/bias/attention per query block.

    The block still attends ALL context keys in one softmax. Large distances
    must be constructed inside the checkpoint, not saved as checkpoint inputs.
    Loop-dependent coordinates are explicit inputs, not late-bound closures.
    """
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


def enable_checkpointing(model):
    """Patch execution only; keep parameters, state keys and model config intact."""
    if getattr(model, "_memory_trial_enabled", False):
        return model
    before = {name: id(value) for name, value in model.state_dict(keep_vars=True).items()}
    if model.mer is not None:
        model.mer.forward = types.MethodType(checkpointed_mer, model.mer)

    def wrap(module):
        if any(getattr(child, "track_running_stats", False) for child in module.modules()):
            raise ValueError("Checkpoint region contains mutable normalization statistics")
        original = module.forward
        def forward(*args, **kwargs):
            if not module.training or not torch.is_grad_enabled():
                return original(*args, **kwargs)
            return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        module.forward = forward

    # Transformer conv51 contains BatchNorm even with norm_name='instance'.
    # Checkpoint only its stateless EPA, leaving BatchNorm and its updates alone.
    regions = [module for module in model.modules() if isinstance(module, EPA)]
    regions += [model.encoder1, model.decoder2]
    if model.tep is not None:
        regions.append(model.tep)
    for module in regions:
        wrap(module)
    after = {name: id(value) for name, value in model.state_dict(keep_vars=True).items()}
    if before != after:
        raise AssertionError("Checkpoint execution patch changed model state identities")
    model._memory_trial_enabled = True
    return model


def _compare_grads(reference, trial, *, rtol, atol):
    errors = []
    for (name, left), (other_name, right) in zip(reference.named_parameters(), trial.named_parameters()):
        assert name == other_name
        if left.grad is None or right.grad is None:
            raise AssertionError(f"Missing parameter gradient: {name}")
        torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol, msg=name)
        if not torch.isfinite(right.grad).all():
            raise AssertionError(f"Nonfinite parameter gradient: {name}")
        errors.append(float((left.grad - right.grad).abs().max()))
    return max(errors, default=0.)


def mer_equivalence():
    torch.manual_seed(123)
    reference = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3, query_chunk_size=5).double().train()
    trial = deepcopy(reference)
    trial.forward = types.MethodType(checkpointed_mer, trial)
    x = torch.randn(2, 4, 3, 4, 6, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = reference(x, [0, 2])
    actual = trial(y, [0, 2])
    torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-12)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-10, atol=1e-12)
    error = _compare_grads(reference, trial, rtol=1e-10, atol=1e-12)
    return {"status": "PASS", "dtype": "float64", "off_center_masters": True,
            "max_parameter_gradient_abs_error": error}


def model_equivalence(device, amp=False):
    torch.manual_seed(321)
    reference = MERITUNETRPP(img_size=(11, 32, 32), feature_size=2, depths=(1, 1, 1, 1)).to(device).train()
    trial = enable_checkpointing(deepcopy(reference))
    assert list(reference.state_dict()) == list(trial.state_dict())
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in trial.parameters())
    x = torch.randn(1, 1, 11, 32, 32, device=device, requires_grad=True)
    target = torch.zeros_like(x)
    target[:, :, 4:7, 5:9, 7:12] = 1
    criterion = MERITLoss()
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    def evaluate(model, image):
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(image, [5])
            losses = criterion(output, target, [5])
        losses["total"].backward()
        return output, losses
    y = x.detach().clone().requires_grad_()
    expected, expected_loss = evaluate(reference, x)
    actual, actual_loss = evaluate(trial, y)
    rtol, atol = (1e-3, 1e-5) if amp else (1e-4, 1e-6)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    for name in expected_loss:
        torch.testing.assert_close(expected_loss[name], actual_loss[name], rtol=rtol, atol=atol)
    torch.testing.assert_close(x.grad, y.grad, rtol=rtol, atol=atol)
    error = _compare_grads(reference, trial, rtol=rtol, atol=atol)
    for name, value in reference.named_buffers():
        torch.testing.assert_close(value, dict(trial.named_buffers())[name], rtol=0, atol=0, msg=name)
    return {"status": "PASS", "amp": amp, "device": str(device), "rtol": rtol, "atol": atol,
            "same_parameter_count_and_state_keys": True, "dropout_rng_preserved": True,
            "normalization_buffers_unchanged": True,
            "max_output_abs_error": max(float((a - b).abs().max().detach()) for a, b in zip(expected, actual)),
            "max_input_gradient_abs_error": float((x.grad - y.grad).abs().max()),
            "max_parameter_gradient_abs_error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=1.)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be in (0,1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    validation = {"status": "FAIL"}
    path = args.output.with_name(args.output.stem + "_equivalence.json")
    if path.exists():
        raise FileExistsError(path)
    try:
        validation["mer"] = mer_equivalence()
        validation["model_fp32"] = model_equivalence(device)
        validation["model_amp"] = model_equivalence(device, amp=True)
        validation["status"] = "PASS"
    except Exception as error:
        validation["error"] = str(error)
        raise
    finally:
        path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print("Output/gradient equivalence: PASS", flush=True)
    if args.validate_only:
        return 0
    gc.collect()
    torch.cuda.empty_cache()
    from merit_covid.engine import preflight
    original_class = preflight.MERITUNETRPP
    def factory(**kwargs):
        return enable_checkpointing(original_class(**kwargs))
    preflight.MERITUNETRPP = factory
    report = preflight.profile_model(args.config, args.output, size=args.size, batch_size=1,
        device="cuda", warmup=1, repeats=1)
    report.update(checkpoint_trial=True, equivalence=validation,
        cloud_allocator_fraction=args.memory_fraction, production_model_source_modified=False,
        execution_regions="MER query blocks including bias; EPA, TEP, encoder1 and final decoder; BatchNorm excluded")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Checkpoint trial:", report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
