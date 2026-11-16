"""Four-step native-input diagnosis: actual SDPA kernels and CPU/GPU hotspots.

Diagnostic execution only. An explicit efficient backend applies to MER alone,
inside each attention call, so checkpoint recomputation uses the same backend.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time
import traceback
from unittest.mock import patch
import warnings

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import collate_slabs
from merit_covid.engine import activation_checkpointing as execution
from merit_covid.engine.preflight import _setup, _sync, _state_hash, _memory
from merit_covid.engine.training import load_experiment, validate_experiment
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss, modules
from tools.merit_real_case_trial import file_hash, verified_case


def event_row(event):
    return dict(name=event.key, count=int(event.count),
        self_cpu_ms=float(event.self_cpu_time_total) / 1000,
        total_cpu_ms=float(event.cpu_time_total) / 1000,
        self_device_ms=float(getattr(event, "self_device_time_total", 0)) / 1000,
        total_device_ms=float(getattr(event, "device_time_total", 0)) / 1000)


def run_profile(config_path, data_dir, specs, output, *, backend="auto", device="cuda"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="NOT_RUN", backend=backend, batches=[], scope="one TRAIN case, one master, four updates",
                  full_training=False, accuracy_evaluated=False)
    case = before = None
    try:
        if backend not in ("auto", "efficient") or (backend == "efficient" and device != "cuda"):
            raise ValueError("Efficient-backend diagnosis requires CUDA")
        spec = specs[0]
        config = load_experiment(config_path)
        config["training"].update(device=device, batch_size=1, num_workers=0)
        config = validate_experiment(config)
        if (config["preprocess"]["size_hw"] is not None
                or config["preprocess"]["intensity_encoding"] != "native"
                or not config["training"]["activation_checkpointing"]):
            raise ValueError("Requires native input and production checkpointing")
        case, before, _ = verified_case(Path(data_dir) / spec["ct_name"], Path(data_dir) / spec["gt_name"],
                                        config["data"]["split_path"], spec)
        sample = CTSlabDataset([case], PreprocessConfig(**config["preprocess"]))[spec["masters"][0]]
        if list(sample["image_slab"].shape) != [1, *config["model"]["img_size"]]:
            raise ValueError("Unexpected native input shape")
        batch = collate_slabs([sample])
        resolved = _setup(config)
        settings = config["training"]
        amp = settings["amp"] and resolved.type == "cuda"
        model = execution.configure_model_execution(MERITUNETRPP(**config["model"]).to(resolved), settings).train()
        model.set_epoch(0, warmup_epochs=settings["warmup_epochs"])
        report.update(case_id=case.case_id, master=spec["masters"][0], torch_version=torch.__version__,
            hardware=torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "CPU",
            initial_state_sha256=_state_hash(model), input_shape=list(batch["image_slab"].shape),
            deterministic=torch.are_deterministic_algorithms_enabled())
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                     weight_decay=settings["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings["amp_init_scale"])
        criterion = MERITLoss(**config["loss"])
        image, truth, valid = (batch[k].to(resolved) for k in ("image_slab", "mask_slab", "valid_region"))
        # Inputs stay on device; these timings diagnose compute, not data loading.
        for name in ("unetr_pp_encoder", "tep", "mer", "encoder1", "decoder5", "decoder4", "decoder3", "decoder2", "pcg"):
            module = getattr(model, name, None)
            if module is None:
                continue
            original = module.forward
            def stage(*args, _original=original, _name=name, **kwargs):
                with torch.profiler.record_function("merit_stage::" + _name):
                    return _original(*args, **kwargs)
            module.forward = stage
        original_attention = modules._mer_attention
        calls = [0]
        def attention(query, keys, values, bias):
            calls[0] += 1
            if "mer_first_call" not in report:
                details = dict(query_shape=list(query.shape), key_shape=list(keys.shape),
                    query_dtype=str(query.dtype), bias_dtype=str(bias.dtype), bias_requires_grad=bias.requires_grad,
                    query_stride=list(query.stride()), key_stride=list(keys.stride()))
                if resolved.type == "cuda":
                    try:
                        mask = bias.to(query.dtype)
                        if mask.ndim == 3:
                            mask = mask.unsqueeze(1)
                        params = torch.backends.cuda.SDPAParams(query.unsqueeze(1), keys.unsqueeze(1),
                            values.unsqueeze(1), mask, 0.0, False, False)
                        with warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter("always")
                            details["efficient_eligible"] = torch.backends.cuda.can_use_efficient_attention(params, debug=True)
                        details["eligibility_warnings"] = [str(w.message) for w in captured]
                    except Exception as error:
                        details["eligibility_probe_error"] = str(error)
                report["mer_first_call"] = details
            with torch.profiler.record_function("merit_stage::mer_sdpa_including_recompute"):
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if backend == "efficient" else nullcontext():
                    return original_attention(query, keys, values, bias)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if resolved.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.reset_peak_memory_stats(resolved)
        with patch.object(modules, "_mer_attention", attention), patch.object(execution, "_mer_attention", attention):
            for index in range(4):
                optimizer.zero_grad(set_to_none=True)
                calls[0] = 0
                profiler = torch.profiler.profile(activities=activities, record_shapes=False,
                    profile_memory=False, with_stack=False) if index == 3 else None
                with profiler if profiler is not None else nullcontext():
                    _sync(resolved)
                    started = time.perf_counter()
                    with torch.profiler.record_function("merit_phase::forward_and_loss"):
                        with torch.autocast(resolved.type, dtype=torch.float16, enabled=amp):
                            prediction = model(image, batch["master_index"], batch["z_offsets_mm"], batch.get("attention_valid"))
                            losses = criterion(prediction, truth, batch["master_index"], valid_region=valid,
                                               z_offsets_mm=batch["z_offsets_mm"])
                    if not all(torch.isfinite(loss).all() for loss in losses.values()):
                        raise FloatingPointError("Nonfinite diagnostic loss")
                    with torch.profiler.record_function("merit_phase::backward"):
                        scaler.scale(losses["total"]).backward()
                    with torch.profiler.record_function("merit_phase::optimizer"):
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                                               error_if_nonfinite=not amp)
                        previous = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                    _sync(resolved)
                    seconds = time.perf_counter() - started
                finite = all(p.grad is not None for p in model.parameters()) and bool(torch.stack([
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
                skipped = scaler.get_scale() < previous
                report["batches"].append(dict(attempt=index + 1, warmup=index == 0, profiled=index == 3,
                    seconds=seconds, gradients_finite=finite, amp_skipped=skipped, mer_calls=calls[0]))
                print(f"Profile {backend} {index+1}/4: {seconds:.3f}s profiled={index == 3} skipped={skipped}", flush=True)
                if not finite or skipped or not torch.isfinite(norm):
                    raise FloatingPointError("Diagnostic optimizer update was not finite and effective")
                del prediction, losses
                if profiler is not None:
                    events = [event_row(e) for e in profiler.key_averages()]
                    report["attention_ops"] = [e for e in events if "scaled_dot_product" in e["name"]]
                    report["stages"] = [e for e in events if e["name"].startswith("merit_")]
                    report["top_self_cpu"] = sorted(events, key=lambda e: e["self_cpu_ms"], reverse=True)[:15]
                    report["top_self_device"] = sorted(events, key=lambda e: e["self_device_ms"], reverse=True)[:15]
                    report["cuda_timing_available"] = any(e["self_device_ms"] > 0 for e in events)
        report.update(status="PASS", two_step_mean_seconds=statistics.mean(r["seconds"] for r in report["batches"]
            if not r["warmup"] and not r["profiled"]), **_memory(resolved))
        report["amp_fp32_retries"] = {n: m._fp32_retry_count for n, m in model.named_modules()
                                     if getattr(m, "_fp32_retry_count", 0)}
    except Exception as error:
        report.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        if case is not None and before is not None:
            report["source_files_unchanged"] = before == {"ct": file_hash(case.image), "gt": file_hash(case.mask)}
            if not report["source_files_unchanged"]:
                report.update(status="FAIL", error="Source data changed")
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data-dir", "specs", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--backend", choices=("auto", "efficient"), default="auto")
    args = parser.parse_args()
    result = run_profile(args.config, args.data_dir, json.loads(args.specs.read_text()), args.output, backend=args.backend)
    print("Runtime profile:", result["status"], flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
