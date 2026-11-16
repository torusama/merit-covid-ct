"""Bounded old/new attention comparison on hash-verified TRAIN CT/GT pairs.

Uses production Dataset/model/loss/execution, with a diagnostic optimizer loop.
It does not instantiate the full-cohort Trainer or evaluate segmentation quality.
Each variant must run in its own process for comparable allocator measurements.
"""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time
import traceback
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import collate_slabs
from merit_covid.data.pipeline.geometry import reorient_to_lps
from merit_covid.engine import activation_checkpointing as execution
from merit_covid.engine.preflight import _setup, _sync, _memory, _state_hash
from merit_covid.engine.training import load_experiment, validate_experiment
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss, MERITUNETRPPAdapter
from merit_covid.models.merit_unetrpp import modules
from tools.merit_real_case_trial import file_hash, verified_case


def legacy_attention(query, keys, values, bias):
    return F.scaled_dot_product_attention(query, keys, values, attn_mask=bias, dropout_p=0.0)


def run_pilot(config_path, data_dir, specs, output, *, variant, device="cuda", rounds=1):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if variant not in ("legacy", "optimized") or type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError("variant must be legacy/optimized and rounds must be 1..3")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="NOT_RUN", variant=variant, device=device, rounds=rounds,
        scope="bounded diagnostic optimizer; production modules; selected TRAIN cases only",
        full_training=False, accuracy_evaluated=False, batches=[], input_checks=[])
    started = time.perf_counter()
    sources = []
    resolved = None
    try:
        if not 1 <= len(specs) <= 3:
            raise ValueError("Select one to three TRAIN cases")
        config = load_experiment(config_path)
        config["training"].update(device=device, batch_size=1, num_workers=0)
        config = validate_experiment(config)
        if (config["preprocess"]["size_hw"] is not None
                or config["preprocess"]["intensity_encoding"] != "native"
                or not config["training"]["activation_checkpointing"]):
            raise ValueError("Requires native XY/intensity and production checkpointing")
        preprocess = PreprocessConfig(**config["preprocess"])
        plan = []
        for spec in specs:
            case, hashes, _ = verified_case(Path(data_dir) / spec["ct_name"],
                Path(data_dir) / spec["gt_name"], config["data"]["split_path"], spec)
            sources.append((case, hashes))
            native_ct = reorient_to_lps(nib.load(case.image)).get_fdata(dtype=np.float32)
            native_gt = reorient_to_lps(nib.load(case.mask)).get_fdata(dtype=np.float32)
            if [11, native_ct.shape[1], native_ct.shape[0]] != config["model"]["img_size"]:
                raise ValueError("Source/model grid mismatch; refusing to resize")
            dataset = CTSlabDataset([case], preprocess)
            if len(spec["masters"]) != 2:
                raise ValueError("Each case needs one positive and one empty master")
            for master, expected_positive in zip(spec["masters"], (True, False)):
                sample = dataset[master]
                indices = sample["slice_indices"].tolist()
                for key, values in (("image_slab", native_ct), ("mask_slab", native_gt)):
                    expected = np.stack([values[:, :, z].T for z in indices])
                    np.testing.assert_array_equal(sample[key][0].numpy(), expected)
                positive = bool(sample["master_mask"].any())
                if positive != expected_positive or sample["metadata"]["lesion_retention"]["resize_applied"]:
                    raise ValueError("Unexpected selected master label or XY resize")
                report["input_checks"].append(dict(case_id=case.case_id, master=master,
                    native_pixels_equal=True, master_positive=positive,
                    slab_positive=bool(sample["mask_slab"].any()),
                    intensity_range=[float(sample["image_slab"].min()), float(sample["image_slab"].max())]))
                plan.append((dataset, master))
            del native_ct, native_gt, sample, expected
        resolved = _setup(config)
        report.update(config=config, torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
            hardware=torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "CPU",
            input_shape=[1, 1, *config["model"]["img_size"]])
        settings = config["training"]
        amp = bool(settings["amp"] and resolved.type == "cuda")
        report["amp"] = amp
        with ExitStack() as stack:
            if variant == "legacy":
                stack.enter_context(patch.object(modules, "_mer_attention", legacy_attention))
                stack.enter_context(patch.object(execution, "_mer_attention", legacy_attention))
            model = execution.configure_model_execution(MERITUNETRPP(**config["model"]).to(resolved), settings).train()
            model.set_epoch(0, warmup_epochs=settings["warmup_epochs"])
            report["initial_state_sha256"] = _state_hash(model)
            report["parameter_count"] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                         weight_decay=settings["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings["amp_init_scale"])
            criterion = MERITLoss(**config["loss"])
            if resolved.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved)
            # The same warmup and sample order are used by both fresh processes.
            schedule = [plan[0]] + plan * rounds
            for index, (dataset, master) in enumerate(schedule):
                load_start = time.perf_counter()
                batch = collate_slabs([dataset[master]])
                data_seconds = time.perf_counter() - load_start
                optimizer.zero_grad(set_to_none=True)
                _sync(resolved)
                step_start = time.perf_counter()
                image = batch["image_slab"].to(resolved)
                truth, valid = batch["mask_slab"].to(resolved), batch["valid_region"].to(resolved)
                with torch.autocast(resolved.type, dtype=torch.float16, enabled=amp):
                    prediction = model(image, batch["master_index"], batch["z_offsets_mm"], batch.get("attention_valid"))
                    losses = criterion(prediction, truth, batch["master_index"],
                        valid_region=valid, z_offsets_mm=batch["z_offsets_mm"])
                if not all(torch.isfinite(value).all() for value in losses.values()):
                    raise FloatingPointError("Nonfinite loss before optimizer update")
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                                      error_if_nonfinite=not amp)
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                _sync(resolved)
                seconds = time.perf_counter() - step_start
                # Extra diagnostic reductions are excluded from the step timer.
                all_gradients = all(p.grad is not None for p in model.parameters()) and bool(torch.stack([
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
                skipped = scaler.get_scale() < previous
                local_master = int(batch["master_index"][0])
                binary = prediction.logits[:, 1] > prediction.logits[:, 0]
                row = dict(attempt=index + 1, warmup=index == 0, case_id=dataset.cases[0].case_id,
                    master=master, seconds=seconds, data_seconds=data_seconds, amp_skipped=skipped,
                    gradients_finite=all_gradients,
                    gradient_norm=float(norm) if torch.isfinite(norm) else None,
                    losses={key: float(value.detach()) for key, value in losses.items()},
                    master_gt_foreground_voxels=int(truth[0, 0, local_master].sum()),
                    master_predicted_foreground_voxels=int(binary[0, local_master].sum()))
                report["batches"].append(row)
                print(f"{variant}: {index+1}/{len(schedule)} {row['case_id']} master={master} "
                      f"step={seconds:.3f}s skipped={skipped}", flush=True)
                if skipped or not all_gradients:
                    raise FloatingPointError("An optimizer attempt was skipped or had missing/nonfinite gradients")
                del prediction, losses, binary, image, truth, valid
            report["final_state_sha256"] = _state_hash(model)
            report["weights_changed"] = report["initial_state_sha256"] != report["final_state_sha256"]
            if not report["weights_changed"]:
                raise AssertionError("Optimizer did not change model weights")
            measured = [row["seconds"] for row in report["batches"] if not row["warmup"]]
            report.update(measured_steps=len(measured), mean_step_seconds=statistics.mean(measured),
                          median_step_seconds=statistics.median(measured), **_memory(resolved))
            report["amp_fp32_retries"] = {name: module._fp32_retry_count for name, module in model.named_modules()
                if getattr(module, "_fp32_retry_count", 0)}
            if variant == "optimized":
                expected_hash = report["final_state_sha256"]
                expected_steps = sorted(int(s["step"].item()) for s in optimizer.state.values() if "step" in s)
                with tempfile.TemporaryDirectory(prefix="mer-attention-checkpoint-") as scratch:
                    checkpoint = Path(scratch) / "diagnostic_only.pt"
                    torch.save(dict(diagnostic_only=True, model=model.state_dict(),
                                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), checkpoint)
                    report["checkpoint_bytes"] = checkpoint.stat().st_size
                    del model, optimizer
                    gc.collect()
                    if resolved.type == "cuda":
                        torch.cuda.empty_cache()
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model = execution.configure_model_execution(MERITUNETRPP(**config["model"]).to(resolved), settings)
                    model.load_state_dict(saved["model"])
                    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                                 weight_decay=settings["weight_decay"])
                    optimizer.load_state_dict(saved["optimizer"])
                    scaler.load_state_dict(saved["scaler"])
                    actual_steps = sorted(int(s["step"].item()) for s in optimizer.state.values() if "step" in s)
                    report["diagnostic_checkpoint"] = dict(model_exact=_state_hash(model) == expected_hash,
                        optimizer_steps_exact=actual_steps == expected_steps, production_resume_tested=False)
                    if not report["diagnostic_checkpoint"]["model_exact"] or actual_steps != expected_steps:
                        raise AssertionError("Diagnostic checkpoint roundtrip mismatch")
                    del saved
                report["diagnostic_checkpoint_deleted"] = True
                case = sources[0][0]
                original_load = nib.load
                def ct_only_load(filename, *args, **kwargs):
                    if Path(filename).resolve() in {c.mask.resolve() for c, _ in sources}:
                        raise AssertionError("CT-only inference opened a GT mask")
                    return original_load(filename, *args, **kwargs)
                model.eval()
                with patch.object(nib, "load", ct_only_load), torch.inference_mode():
                    inference = CTSlabDataset([case], preprocess, require_mask=False)
                    batch = collate_slabs([inference[specs[0]["masters"][0]]])
                    batch["image_slab"] = batch["image_slab"].to(resolved)
                    with torch.autocast(resolved.type, dtype=torch.float16, enabled=amp):
                        logits = MERITUNETRPPAdapter(model).predict_logits(batch)
                    if logits.shape != (1, 1, *config["model"]["img_size"][1:]) or not torch.isfinite(logits).all():
                        raise FloatingPointError("Invalid CT-only master logits")
                    report["inference"] = dict(ct_only_guard="PASS", shape=list(logits.shape),
                        full_volume_reconstructed=False)
            report["status"] = "PASS"
    except Exception as error:
        report.update(status="OOM" if isinstance(error, torch.OutOfMemoryError) else "FAIL",
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        unchanged = all({"ct": file_hash(case.image), "gt": file_hash(case.mask)} == before for case, before in sources)
        report["source_files_unchanged"] = unchanged
        if not unchanged:
            report.update(status="FAIL", error="Source data changed during pilot")
        report["total_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data-dir", "specs", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--variant", choices=("legacy", "optimized"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    report = run_pilot(args.config, args.data_dir, json.loads(args.specs.read_text()), args.output,
                       variant=args.variant, device=args.device, rounds=args.rounds)
    print("Pilot result:", report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
