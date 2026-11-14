"""Bounded native CT/GT pilot using shared Dataset/model/loss, not full training.

Only the uploaded, hash-verified TRAIN case is opened. The canonical split is
checked without fabricating paths/files for the missing 198 cases. No patient
data or large model checkpoint is copied into the report directory.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

if __package__:
    from .merit_checkpoint_trial import enable_checkpointing, mer_equivalence, model_equivalence
else:
    from merit_checkpoint_trial import enable_checkpointing, mer_equivalence, model_equivalence

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.data.pipeline.geometry import reorient_to_lps
from merit_covid.engine.training import load_experiment, validate_experiment
from merit_covid.engine.preflight import _setup, _sync, _memory
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITUNETRPPAdapter, MERITLoss


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verified_case(ct, mask, split_path, spec):
    ct, mask, split_path = Path(ct).resolve(), Path(mask).resolve(), Path(split_path)
    if ct == mask:
        raise ValueError("CT and mask must be distinct source files")
    raw = split_path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != spec["split_sha256_lf"]:
        raise ValueError("Canonical split bytes do not match the frozen specification")
    split = json.loads(raw)
    if split["manifest_fingerprint"] != spec["manifest_fingerprint"]:
        raise ValueError("Canonical manifest identity mismatch")
    part = next(row for row in split["folds"] if row["fold"] == spec["fold"])
    if (spec["case_id"] not in part["train"]["case_ids"]
            or spec["group_id"] not in part["train"]["group_ids"]
            or any(spec["case_id"] in part[key]["case_ids"] or spec["group_id"] in part[key]["group_ids"]
                   for key in ("validation", "test"))):
        raise ValueError("Uploaded case/group must belong only to the selected TRAIN partition")
    hashes = {"ct": file_hash(ct), "gt": file_hash(mask)}
    if hashes != spec["sha256"]:
        raise ValueError(f"Uploaded CT/GT bytes do not match the expected case: {hashes}")
    for source in (ct, mask):
        if list(nib.load(source).shape) != spec["shape_xyz"]:
            raise ValueError("Source shape does not match the native case specification")
    return Case(spec["case_id"], ct, mask, spec["group_id"]), hashes, split["identity"]


def verify_sample(sample, ct, mask):
    indices = sample["slice_indices"].tolist()
    image = np.stack([np.asarray(ct.dataobj[:, :, i], dtype=np.float32).T for i in indices])
    truth = np.stack([np.asarray(mask.dataobj[:, :, i], dtype=np.float32).T for i in indices])
    np.testing.assert_array_equal(sample["image_slab"][0].numpy(), image)
    np.testing.assert_array_equal(sample["mask_slab"][0].numpy(), truth)
    info = sample["metadata"]
    if (info["lesion_retention"]["resize_applied"] or info["z_resampled"]
            or info["processed_size_hw"] != info["native_size_hw"]):
        raise AssertionError("Native input was resized or resampled")
    return {"master": int(sample["master_slice_index"]), "selected_slices": indices,
            "ct_gt_equal_native_source": True,
            "positive_voxels_per_selected_slice": (truth == 1).sum((1, 2)).tolist(),
            "presence_per_selected_slice": (truth == 1).any((1, 2)).tolist(),
            "ct_range": [float(image.min()), float(image.max())]}


def run_trial(config_path, ct, mask, split_path, spec, output, *, steps=2,
              device="cuda", memory_fraction=1., check_equivalence=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "NOT_RUN", "scope": "one verified TRAIN case, bounded native optimizer attempts",
              "requested_attempts": steps, "optimizer_steps": 0, "amp_skipped_steps": 0,
              "full_training": False, "accuracy_evaluated": False, "checkpoint_saved": False,
              "production_source_modified": False, "input_checks": [], "batches": []}
    pilot_start = time.perf_counter()
    resolved, case, before = None, None, None
    try:
        if type(steps) is not int or not 1 <= steps <= 8:
            raise ValueError("Pilot attempts must be between 1 and 8")
        if not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0,1]")
        case, before, identity = verified_case(ct, mask, split_path, spec)
        report.update(case_id=case.case_id, fold=spec["fold"], identity=identity,
            source_sha256_before=before, split_sha256_lf=spec["split_sha256_lf"])
        config = load_experiment(config_path)
        if config["preprocess"]["intensity_encoding"] != "native" or config["preprocess"]["hu_window"] is not None:
            raise ValueError("This pilot requires native intensity without a HU window")
        if config["preprocess"]["sampling_mode"] != "index" or config["preprocess"]["slab_size"] != 11:
            raise ValueError("This pilot requires the existing 11-slice index sampler")
        source_ct, source_gt = reorient_to_lps(nib.load(case.image)), reorient_to_lps(nib.load(case.mask))
        native_hw = [source_ct.shape[1], source_ct.shape[0]]
        config["preprocess"]["size_hw"] = native_hw
        config["model"]["img_size"] = [11, *native_hw]
        config["training"].update(device=device, batch_size=1, num_workers=0)
        config = validate_experiment(config)
        resolved = _setup(config)
        if resolved.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction),
                resolved.index if resolved.index is not None else torch.cuda.current_device())
        report.update(config=config, torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
            hardware=torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "CPU",
            native_shape_xyz=list(source_ct.shape), native_size_hw=native_hw,
            cloud_allocator_fraction=memory_fraction)
        input_start = time.perf_counter()
        dataset = CTSlabDataset([case], PreprocessConfig(**config["preprocess"]))
        masters = [(len(dataset) // 2 + i) % len(dataset) for i in range(steps)]
        def forbidden_resize(*args, **kwargs):
            raise AssertionError("Native pilot attempted XY interpolation")
        with patch("merit_covid.data.pipeline.preprocessing.resize_in_plane", forbidden_resize):
            for master in masters:
                sample = dataset[master]
                report["input_checks"].append(verify_sample(sample, source_ct, source_gt))
                if "geometry" not in report:
                    report["geometry"] = {key: sample["metadata"][key] for key in (
                        "original_orientation", "canonical_orientation", "original_shape_xyz", "canonical_shape_xyz",
                        "original_affine_mm", "canonical_affine_mm", "spatial_unit_in_header", "assume_mm",
                        "nifti_scaling", "intensity_encoding", "hu_window") if key in sample["metadata"]}
        report.update(master_indices=masters, input_shape=[1, 1, 11, *native_hw],
            xy_interpolation_called=False, normalization_applied=False,
            dataset_and_native_comparison_seconds=time.perf_counter() - input_start)
        if check_equivalence:
            report["equivalence"] = {"mer": mer_equivalence(), "fp32": model_equivalence(resolved)}
            if resolved.type == "cuda":
                report["equivalence"]["amp"] = model_equivalence(resolved, amp=True)
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
        _setup(config)  # Restore the experiment seed after the independent comparisons.
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
        settings = config["training"]
        model = enable_checkpointing(MERITUNETRPP(**config["model"]).to(resolved)).train()
        model.set_epoch(0, warmup_epochs=settings["warmup_epochs"])
        report.update(parameter_count=sum(p.numel() for p in model.parameters()),
                      actual_pcg_ramp=float(model.pcg_ramp))
        criterion = MERITLoss(**config["loss"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        amp = settings["amp"] and resolved.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings["amp_init_scale"])
        loader = make_dataloader(torch.utils.data.Subset(dataset, masters), batch_size=1,
            shuffle=False, num_workers=0, seed=settings["seed"])
        report["parameter_sha256_before"] = parameter_hash(model)
        with patch("merit_covid.data.pipeline.preprocessing.resize_in_plane", forbidden_resize):
            for i, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                _sync(resolved)
                start = time.perf_counter()
                image, truth = batch["image_slab"].to(resolved), batch["mask_slab"].to(resolved)
                with torch.autocast(resolved.type, dtype=torch.float16, enabled=amp):
                    prediction = model(image, batch["master_index"], batch["z_offsets_mm"])
                    losses = criterion(prediction, truth, batch["master_index"],
                        valid_region=batch["valid_region"].to(resolved), z_offsets_mm=batch["z_offsets_mm"])
                if not all(torch.isfinite(value).all() for value in losses.values()):
                    raise FloatingPointError("Nonfinite real-CT loss before optimizer update")
                scaler.scale(losses["total"]).backward()
                if any(p.grad is None for p in model.parameters()):
                    raise FloatingPointError("Missing real-CT parameter gradients")
                if not amp and any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite real-CT gradients")
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=not amp)
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                _sync(resolved)
                skipped = scaler.get_scale() < previous
                report["optimizer_steps"] += int(not skipped)
                report["amp_skipped_steps"] += int(skipped)
                row = {"attempt": i + 1, "master": masters[i], "seconds": time.perf_counter() - start,
                       "skipped": skipped, "gradient_norm_finite": bool(torch.isfinite(norm)),
                       "losses": {key: float(value.detach()) for key, value in losses.items()}}
                report["batches"].append(row)
                if not skipped and not row["gradient_norm_finite"]:
                    raise FloatingPointError("Effective optimizer attempt had a nonfinite gradient norm")
                print(f"Real CT attempt {i+1}/{steps}: master={masters[i]}, seconds={row['seconds']:.3f}, skipped={skipped}", flush=True)
                del prediction, losses, image, truth
        report["parameter_sha256_after"] = parameter_hash(model)
        report["weights_changed"] = report["parameter_sha256_before"] != report["parameter_sha256_after"]
        if report["optimizer_steps"] != steps or not report["weights_changed"]:
            raise FloatingPointError("Requested effective optimizer updates were not all completed")
        optimizer.zero_grad(set_to_none=True)
        # One CT-only master prediction, not an incomplete full-volume export.
        model.eval()
        original_load = nib.load
        def ct_only_load(filename, *args, **kwargs):
            if Path(filename).resolve() == case.mask.resolve():
                raise AssertionError("CT-only inference attempted to open GT")
            return original_load(filename, *args, **kwargs)
        with patch.object(nib, "load", ct_only_load), torch.inference_mode():
            inference = CTSlabDataset([case], dataset.config, require_mask=False)
            batch = next(iter(make_dataloader(torch.utils.data.Subset(inference, [masters[0]]))))
            batch["image_slab"] = batch["image_slab"].to(resolved)
            logits = MERITUNETRPPAdapter(model).predict_logits(batch)
            if logits.shape != (1, 1, *native_hw) or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid CT-only native master logits")
            probability = logits.float().sigmoid()
            report["inference"] = {"ct_only_guard": "PASS", "master_only": True,
                "full_volume_reconstructed": False, "shape": list(logits.shape),
                "probability_range": [float(probability.min()), float(probability.max())]}
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="OOM" if isinstance(error, torch.OutOfMemoryError) else "FAIL",
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if case is not None and before is not None:
            after = {"ct": file_hash(case.image), "gt": file_hash(case.mask)}
            report.update(source_sha256_after=after, source_files_unchanged=after == before)
            if after != before:
                report.update(status="FAIL", error="Source files changed during pilot")
        if resolved is not None:
            report.update(_memory(resolved))
        report["total_pilot_seconds"] = time.perf_counter() - pilot_start
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "ct", "mask", "split", "spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=1.)
    args = parser.parse_args()
    report = run_trial(args.config, args.ct, args.mask, args.split,
        json.loads(args.spec.read_text()), args.output, steps=args.steps, memory_fraction=args.memory_fraction)
    print("Real CT pilot:", report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
