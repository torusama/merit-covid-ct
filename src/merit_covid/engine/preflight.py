"""Bounded optimizer profiling and real-CT integration checks, not evaluation.

Commands run separately so a failed large-grid allocation cannot contaminate
another measurement. Production YAML, source images and splits are never edited.
"""

from copy import deepcopy
import gc
import hashlib
import math
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import collate_slabs
from merit_covid.data.pipeline.preflight import dump_json
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
from .slab_inference import predict_slab_case
from .training import (load_experiment, validate_experiment, resolve_device,
                       load_development_split, Trainer)


def diagnostic_config(path, size, device, batch_size=1):
    config = deepcopy(load_experiment(path))
    if type(size) is not int or size < 32 or size % 32:
        raise ValueError("Size must be a positive multiple of 32")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive batch size required")
    # Preserve a native config when profiling its declared grid. Other sizes
    # are explicitly requested diagnostic ablations, not a production default.
    if config["preprocess"]["size_hw"] is not None or config["model"]["img_size"][1:] != [size, size]:
        config["preprocess"]["size_hw"] = [size, size]
    config["model"]["img_size"] = [11, size, size]
    config["training"].update(device=device, batch_size=batch_size, num_workers=0, shuffle_mode="sequential")
    if config["preprocess"]["sampling_mode"] != "index" or config["preprocess"]["slab_size"] != 11:
        raise ValueError("This preflight requires 11 index-sampled slices")
    return validate_experiment(config)


def _setup(config):
    settings = config["training"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(settings["deterministic"])
    torch.backends.cudnn.benchmark = False
    return resolve_device(settings["device"])


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device):
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _failure(report, error):
    report.update(status="OOM" if isinstance(error, torch.OutOfMemoryError) else "FAIL",
                  error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())


def check_native_input(config_path, output, *, fold=0, case_id="volume-covid19-A-0025"):
    """Verify one full native TRAIN slab against source pixels, with no model run.

    Fail instead of resizing a source that does not match the declared grid.
    This is an input-integrity check, not a cohort audit or a training result.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "NOT_RUN", "case_id": case_id, "fold": fold,
              "scope": "one TRAIN case, one central full-native slab; no model or optimizer"}
    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    try:
        config = load_experiment(config_path)
        preprocess = PreprocessConfig(**config["preprocess"])
        if preprocess.intensity_encoding != "native" or preprocess.hu_window is not None:
            raise ValueError("Native input check requires native intensity without a HU window")
        cases, split, _, split_hash = load_development_split(config)
        part = next(f for f in split["folds"] if f["fold"] == fold)
        if case_id not in part["train"]["case_ids"]:
            raise ValueError("Native input check must use the selected fold TRAIN partition")
        case = next(c for c in cases if c.case_id == case_id)
        from merit_covid.data.pipeline.geometry import reorient_to_lps
        ct, truth = reorient_to_lps(nib.load(case.image)), reorient_to_lps(nib.load(case.mask))
        shape = (ct.shape[1], ct.shape[0])
        if (preprocess.size_hw is not None and shape != tuple(preprocess.size_hw)
                or shape != tuple(config["model"]["img_size"][1:])):
            raise ValueError("Source native XY does not match the declared grid; refusing to resize")
        before = {"ct": file_hash(case.image), "gt": file_hash(case.mask)}
        dataset = CTSlabDataset([case], preprocess)
        master = len(dataset) // 2
        def forbidden_resize(*args, **kwargs):
            raise AssertionError("Native route unexpectedly attempted XY interpolation")
        with patch("merit_covid.data.pipeline.preprocessing.resize_in_plane", forbidden_resize):
            sample = dataset[master]
        indices = sample["slice_indices"].tolist()
        native_ct = np.stack([np.asarray(ct.dataobj[:, :, i], dtype=np.float32).T for i in indices])
        native_gt = np.stack([np.asarray(truth.dataobj[:, :, i], dtype=np.float32).T for i in indices])
        np.testing.assert_array_equal(sample["image_slab"][0].numpy(), native_ct)
        np.testing.assert_array_equal(sample["mask_slab"][0].numpy(), native_gt)
        info = sample["metadata"]
        if info["lesion_retention"]["resize_applied"]:
            raise AssertionError("Native input recorded a resize")
        after = {"ct": file_hash(case.image), "gt": file_hash(case.mask)}
        if before != after:
            raise AssertionError("Source CT/GT bytes changed")
        report.update(status="PASS", config=config, split_sha256_lf=split_hash,
            canonical_shape_xyz=list(ct.shape), input_shape=list(sample["image_slab"].shape),
            original_orientation=list(nib.aff2axcodes(nib.load(case.image).affine)),
            master_slice_index=master, slice_indices=indices,
            native_size_hw=list(shape), processed_size_hw=info["processed_size_hw"],
            xy_interpolation_called=False, crop_or_tiling_applied=False,
            ct_equals_native_source=True, gt_equals_native_source=True,
            positive_voxels_by_selected_slice=(native_gt == 1).sum(axis=(1, 2)).tolist(),
            presence_by_selected_slice=(native_gt == 1).any(axis=(1, 2)).tolist(),
            ct_range=[float(native_ct.min()), float(native_ct.max())],
            source_sha256_before=before, source_sha256_after=after,
            source_files_unchanged=True, lesion_retention=info["lesion_retention"],
            model_run=False, optimizer_steps=0)
    except Exception as error:
        _failure(report, error)
    dump_json(output, report)
    return report


def profile_model(config_path, output, *, size, batch_size, device="cuda", warmup=1, repeats=2,
                  memory_fraction=None):
    """Full training step including AdamW state; synthetic inputs, not accuracy."""
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("At least one warmup and measured iteration required")
    if memory_fraction is not None and (type(memory_fraction) not in (int, float)
            or not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1):
        raise ValueError("memory_fraction must be finite and in (0,1]")
    if memory_fraction is not None and torch.device(device).type != "cuda":
        raise ValueError("memory_fraction requires the isolated CUDA profiler")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "NOT_RUN", "size": size, "batch_size": batch_size,
              "warmup": warmup, "repeats": repeats, "input": "synthetic normal intensity, binary central lesion",
              "scope": "forward + MERIT loss + backward + unscale/clip + AdamW optimizer step; CUDA AMP as configured",
              "timings_seconds": [], "optimizer_steps": 0, "amp_skipped_steps": 0,
              "cuda_memory_fraction": memory_fraction}
    resolved = None
    try:
        config = diagnostic_config(config_path, size, device, batch_size)
        report["config"] = config
        resolved = _setup(config)
        settings = config["training"]
        report.update(device=str(resolved), torch_version=torch.__version__, cuda_runtime=torch.version.cuda)
        if resolved.type == "cuda":
            if memory_fraction is not None:
                # This command runs in its own process; reserve room for display
                # use instead of letting WDDM spill a large job into host RAM.
                index = resolved.index if resolved.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(float(memory_fraction), index)
            free, total = torch.cuda.mem_get_info(resolved)
            report.update(hardware=torch.cuda.get_device_name(resolved),
                          total_vram_mib=torch.cuda.get_device_properties(resolved).total_memory / 2**20,
                          free_vram_before_model_mib=free / 2**20)
            torch.cuda.reset_peak_memory_stats(resolved)
        model = MERITUNETRPP(**config["model"]).to(resolved).train()
        from .activation_checkpointing import configure_model_execution
        configure_model_execution(model, settings)
        model.set_epoch(0, warmup_epochs=settings["warmup_epochs"])
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        criterion = MERITLoss(**config["loss"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        amp = settings["amp"] and resolved.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings["amp_init_scale"])
        image = torch.randn(batch_size, 1, 11, size, size, device=resolved)
        target = torch.zeros(batch_size, 1, 11, size, size, device=resolved)
        target[:, :, :, size//3:2*size//3, size//3:2*size//3] = 1
        master = torch.full((batch_size,), 5, dtype=torch.long)
        offsets = torch.arange(-5, 6).float().repeat(batch_size, 1)
        valid = torch.ones_like(target, dtype=torch.bool)
        for iteration in range(warmup + repeats):
            optimizer.zero_grad(set_to_none=True)
            _sync(resolved)
            start = time.perf_counter()
            with torch.autocast(device_type=resolved.type, dtype=torch.float16, enabled=amp):
                predictions = model(image, master, offsets)
                losses = criterion(predictions, target, master, valid_region=valid, z_offsets_mm=offsets)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError("Non-finite profile loss")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            _sync(resolved)
            elapsed = time.perf_counter() - start
            skipped = scaler.get_scale() < previous_scale
            report["amp_skipped_steps"] += int(skipped)
            report["optimizer_steps"] += int(not skipped)
            if iteration >= warmup:
                report["timings_seconds"].append(elapsed)
            report["last_losses"] = {key: float(value.detach()) for key, value in losses.items()}
            report["last_gradient_norm_finite"] = bool(torch.isfinite(norm))
            print(f"profile {size} batch={batch_size} iteration={iteration+1}/{warmup+repeats} seconds={elapsed:.3f} skipped={skipped}", flush=True)
            del predictions, losses
        report.update(status="PASS" if report["optimizer_steps"] == warmup+repeats else "FAIL",
                      mean_step_seconds=float(np.mean(report["timings_seconds"])),
                      seconds_per_slab=float(np.mean(report["timings_seconds"]))/batch_size)
    except Exception as error:
        _failure(report, error)
    finally:
        if resolved is not None:
            report.update(_memory(resolved))
        dump_json(output, report)
    return report


def pilot_case(config_path, output, *, fold=0, case_id="volume-covid19-A-0025", size=512, steps=4, device="cuda"):
    """Small real TRAIN-case run through the production Trainer and inference.

    No validation/test images, threshold tuning or generalization claims. The
    output checkpoint is diagnostic only and must not be used as a best model.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive bounded step count required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "NOT_RUN", "case_id": case_id, "fold": fold, "size": size,
              "requested_steps": steps, "scope": "Single fold-training case only; no quality assessment",
              "normalization": "Production config unchanged; native intensity, no HU window"}
    resolved = None
    start = time.perf_counter()
    try:
        config = diagnostic_config(config_path, size, device)
        resolved = _setup(config)
        cases, split, identity, split_hash = load_development_split(config)
        part = next(f for f in split["folds"] if f["fold"] == fold)
        case = next(c for c in cases if c.case_id == case_id)
        if case_id not in part["train"]["case_ids"]:
            raise ValueError("Pilot case must belong to the selected fold TRAIN partition")
        dataset = CTSlabDataset([case], PreprocessConfig(**config["preprocess"]))
        # Central consecutive masters, not selected by label or intensity.
        indices = [(len(dataset)//2 + i) % len(dataset) for i in range(steps)]
        loader = DataLoader(dataset, batch_size=1, sampler=indices, num_workers=0,
                            collate_fn=collate_slabs, generator=torch.Generator().manual_seed(config["training"]["seed"]))
        validation = [c for c in cases if c.case_id in part["validation"]["case_ids"]]
        provenance = {"fold": fold, "split_fingerprint": split["manifest_fingerprint"],
            "split_sha256_lf": split_hash, "identity": identity, "partitions": part,
            "all_case_ids": [c.case_id for c in cases], "all_group_ids": sorted({c.group_id for c in cases}),
            "all_image_paths": sorted(str(c.image.resolve()) for c in cases),
            "pilot_only": True, "pilot_case_id": case_id, "master_indices": indices}
        report.update(config=config, provenance=provenance, master_indices=indices,
                      hardware=torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "CPU")
        print(f"pilot: {case_id}, {steps} train steps, 11 index slices, native intensity", flush=True)
        trainer = Trainer(config, loader, validation, provenance, output)
        before = _state_hash(trainer.model)
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
        _sync(resolved)
        training_start = time.perf_counter()
        report["train"] = trainer.train_epoch(0)
        _sync(resolved)
        report["train_seconds"] = time.perf_counter() - training_start
        report["final_gradients_finite"] = all(bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters() if p.grad is not None)
        after = _state_hash(trainer.model)
        report["weights_changed"] = before != after
        if not report["weights_changed"] or not report["final_gradients_finite"] or report["train"]["optimizer_steps"] < 1:
            raise FloatingPointError("Pilot did not produce a finite, effective optimizer update")
        checkpoint = output / "pilot_only.pt"
        trainer.save_checkpoint(checkpoint, 0)
        expected_steps = sorted(int(state["step"].item()) for state in trainer.optimizer.state.values() if "step" in state)
        del trainer
        gc.collect()
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
        print("pilot: checkpoint saved; checking fresh Trainer resume", flush=True)
        resumed = Trainer(config, loader, validation, provenance, output)
        resumed.resume(checkpoint)
        actual_steps = sorted(int(state["step"].item()) for state in resumed.optimizer.state.values() if "step" in state)
        report["checkpoint"] = {"path": str(checkpoint.resolve()), "model_exact_match": _state_hash(resumed.model) == after,
            "optimizer_step_counters_match": expected_steps == actual_steps and bool(expected_steps),
            "start_epoch": resumed.start_epoch, "diagnostic_only": True}
        if not report["checkpoint"]["model_exact_match"] or not report["checkpoint"]["optimizer_step_counters_match"] or resumed.start_epoch != 1:
            raise AssertionError("Checkpoint roundtrip failed")
        model = resumed.model.eval()
        del resumed, loader, dataset
        gc.collect()
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
        print("pilot: CT-only inference over every native master slice", flush=True)
        original_load = nib.load
        def ct_only_load(filename, *args, **kwargs):
            if Path(filename).resolve() == case.mask.resolve():
                raise AssertionError("Inference attempted to read GT")
            return original_load(filename, *args, **kwargs)
        inference_start = time.perf_counter()
        with patch.object(nib, "load", ct_only_load):
            prediction = predict_slab_case(model, case, PreprocessConfig(**config["preprocess"]), batch_size=1)
        _sync(resolved)
        native = prediction.to_original_xyz()
        source = nib.load(str(case.image))
        if native.shape != source.shape or not np.isfinite(native).all() or native.min() < 0 or native.max() > 1:
            raise AssertionError("Native reconstruction shape/range check failed")
        np.testing.assert_allclose(prediction.metadata["original_affine"], source.affine)
        # Export an explicitly diagnostic binary prediction with the ORIGINAL
        # header/affine, then read it back to verify the on-disk geometry too.
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        restored = nib.Nifti1Image((native >= config["training"]["threshold"]).astype(np.uint8), source.affine, header)
        export = output / "diagnostic_prediction_original.nii.gz"
        nib.save(restored, export)
        reloaded = nib.load(str(export))
        np.testing.assert_allclose(reloaded.affine, source.affine)
        assert reloaded.shape == source.shape
        report["inference"] = {"ct_only_guard": "PASS", "native_reconstruction": "PASS",
            "canonical_shape_zyx": list(prediction.probability_zyx.shape), "original_shape_xyz": list(native.shape),
            "probability_min": float(native.min()), "probability_max": float(native.max()),
            "original_affine_match": True, "export": str(export.resolve()),
            "seconds": time.perf_counter()-inference_start, "quality": "NOT_EVALUATED; diagnostic checkpoint only"}
        report["status"] = "PASS"
    except Exception as error:
        _failure(report, error)
    finally:
        if resolved is not None:
            report.update(_memory(resolved))
        report["seconds"] = time.perf_counter()-start
        dump_json(output / "pilot.json", report)
    return report
