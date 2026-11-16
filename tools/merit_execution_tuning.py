"""Isolated full-context MER chunk/recomputation experiments, with no protocol edits."""
import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import types
from unittest.mock import patch

import torch

from merit_covid.engine import activation_checkpointing as execution
from merit_covid.models.merit_unetrpp import MasterEvidenceRouting
from tools import merit_factorized_attention as factor


@contextmanager
def deterministic_mode(enabled):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(enabled)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@contextmanager
def recomputation(enabled):
    def direct(function, *args, **kwargs):
        kwargs.pop("use_reentrant", None)
        kwargs.pop("preserve_rng_state", None)
        return function(*args, **kwargs)
    with nullcontext() if enabled else patch.object(factor, "checkpoint", direct):
        yield


def compare(chunk, recompute, *, native=False, amp=True, masked=False, device="cuda", fast_reductions=False):
    """Check actual outputs and all gradients; native uses the real MER token grid."""
    torch.manual_seed(842)
    channels, depth, height, width = (16, 11, 128, 128) if native else (4, 3, 5, 6)
    reference = MasterEvidenceRouting(channels, attention_dim=64 if native else 8,
        gate_hidden=16 if native else 3, query_chunk_size=512 if native else 7,
        position_mode="physical_mm").to(device).train()
    candidate = deepcopy(reference)
    candidate.query_chunk_size = chunk
    for model in (reference, candidate):
        model.forward = types.MethodType(factor.factorized_mer, model)
    x = torch.randn(1, channels, depth, height, width, device=device)
    offsets = (torch.arange(depth, dtype=torch.float32)[None] - depth // 2) * 2.5
    mask = torch.ones(1, depth, dtype=torch.bool) if masked else None
    if masked:
        mask[:, :depth // 2] = False
    results = []
    for model, enabled in ((reference, True), (candidate, recompute)):
        xi = x.detach().clone().requires_grad_()
        # The reference retains deterministic reductions. The optional candidate
        # permits parallel reductions; verify its numerical differences explicitly.
        with recomputation(enabled), deterministic_mode(not fast_reductions or model is reference):
            with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=amp):
                output = model(xi, [depth // 2], offsets, mask)
                loss = output.square().mean()
            loss.backward()
        results.append((output.detach().cpu(), xi.grad.detach().cpu(),
            {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}))
    rtol, atol = (5e-3, 2e-6) if amp else (1e-4, 1e-7)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=rtol, atol=atol)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=rtol, atol=atol)
    for name, value in results[0][2].items():
        assert torch.isfinite(results[1][2][name]).all(), name
        torch.testing.assert_close(value, results[1][2][name], rtol=rtol, atol=atol, msg=name)
    return dict(status="PASS", chunk=chunk, recompute=recompute, native_token_grid=native,
        amp=amp, masked=masked, fast_reductions=fast_reductions, rtol=rtol, atol=atol,
        output_max_abs_diff=float((results[0][0]-results[1][0]).abs().max()),
        input_grad_max_abs_diff=float((results[0][1]-results[1][1]).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, choices=(512, 1024, 4096, 16384), required=True)
    parser.add_argument("--recompute", choices=("on", "off"), required=True)
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--fast-reductions", action="store_true")
    for name in ("config", "data-dir", "specs"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.guard:
        # Compare at the actual MER grid before a full-model native CT worker.
        result = compare(args.chunk, args.recompute == "on", native=True, fast_reductions=args.fast_reductions)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("Native MER guard:", result, flush=True)
        return
    from tools import merit_attention_pilot as pilot
    original_setup = pilot._setup
    def setup(config):
        if args.fast_reductions:
            config["training"]["deterministic"] = False
        return original_setup(config)
    original = execution.configure_model_execution
    def configure(model, settings):
        model = factor.enable_factorized_mer(original(model, settings))
        model.mer.query_chunk_size = args.chunk
        return model
    with patch.object(pilot, "_setup", setup), patch.object(execution, "configure_model_execution", configure), recomputation(args.recompute == "on"):
        result = pilot.run_pilot(args.config, args.data_dir,
            json.loads(args.specs.read_text(encoding="utf-8")), args.output,
            variant="optimized", rounds=1)
    print("Execution tuning:", args.chunk, args.recompute, result["status"], flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
