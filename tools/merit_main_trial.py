"""Bounded real-input check of the integrated production Trainer, not a full run.

Requires real sources for the complete manifest. Selects one TRAIN volume for
optimizer updates and one VALIDATION case for constructor/header checks only.
No monkeypatch of model execution, optimizer loop or source validation.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
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
from merit_covid.data.pipeline.geometry import reorient_to_lps
from merit_covid.engine.training import (
    Trainer, load_experiment, load_development_split, validate_experiment, resolve_device,
)
from merit_covid.models.merit_unetrpp import MERITUNETRPPAdapter


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_trial(config_path, manifest, spec, output, scratch, *, steps=2, device="cuda", expected_cases=199):
    output, scratch = Path(output), Path(scratch)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = dict(status="NOT_RUN", scope="integrated Trainer, one TRAIN case, bounded steps and diagnostic resume",
                  full_training=False, validation_evaluated=False, accuracy_evaluated=False,
                  production_source_modified=False, checkpoint_in_report_zip=False)
    before, case, resolved = None, None, None
    started = time.perf_counter()
    try:
        if type(steps) is not int or not 1 <= steps <= 8:
            raise ValueError("steps must be in 1..8")
        config = load_experiment(config_path)
        config["data"]["manifest"] = str(Path(manifest).resolve())
        config["training"].update(device=device, batch_size=1, num_workers=0, shuffle_mode="sequential")
        config = validate_experiment(config)
        resolved = resolve_device(config["training"]["device"])
        preprocess = PreprocessConfig(**config["preprocess"])
        if (preprocess.size_hw is not None or preprocess.intensity_encoding != "native"
                or preprocess.hu_window is not None or preprocess.assume_mm
                or not config["training"]["activation_checkpointing"]):
            raise ValueError("Requires native XY/intensity, strict source units and integrated checkpointing")
        cases, split, identity, split_hash = load_development_split(config)
        if len(cases) != expected_cases or split_hash != spec["split_sha256_lf"]:
            raise ValueError("Cohort count or canonical split bytes differ from specification")
        if split["manifest_fingerprint"] != spec["manifest_fingerprint"]:
            raise ValueError("Manifest identity differs from canonical split")
        part = next(row for row in split["folds"] if row["fold"] == spec["fold"])
        case = next(row for row in cases if row.case_id == spec["case_id"])
        if (case.case_id not in part["train"]["case_ids"] or case.group_id != spec["group_id"]
                or case.group_id not in part["train"]["group_ids"]):
            raise ValueError("Requested source must belong to the frozen TRAIN partition")
        before = {"ct": file_hash(case.image), "gt": file_hash(case.mask)}
        if before != spec["sha256"]:
            raise ValueError("TRAIN CT/GT hashes differ from the expected originals")
        native_ct, native_gt = reorient_to_lps(nib.load(case.image)), reorient_to_lps(nib.load(case.mask))
        if list(nib.load(case.image).shape) != spec["shape_xyz"]:
            raise ValueError("TRAIN source shape differs from specification")
        dataset = CTSlabDataset([case], preprocess)
        masters = [(len(dataset) // 2 + i) % len(dataset) for i in range(steps + 1)]
        sample = dataset[masters[0]]
        indices = sample["slice_indices"].tolist()
        for key, image in (("image_slab", native_ct), ("mask_slab", native_gt)):
            truth = np.stack([np.asarray(image.dataobj[:, :, z], dtype=np.float32).T for z in indices])
            np.testing.assert_array_equal(sample[key][0].numpy(), truth)
        if sample["metadata"]["lesion_retention"]["resize_applied"]:
            raise ValueError("Unexpected XY resize")
        # One explicitly declared, distinct validation member, headers only.
        # Remaining sources are bound by production Trainer path/stat checks;
        # neither their units nor their voxels are claimed as fully audited.
        validation_id = part["validation"]["case_ids"][0]
        validation = [next(row for row in cases if row.case_id == validation_id)]
        report.update(case_id=case.case_id, fold=spec["fold"], identity=identity,
            config=config, input_shape=[1, *sample["image_slab"].shape], master_indices=masters,
            input_matches_native=True, xy_interpolation_called=False, normalization_applied=False,
            sources_available=len(cases), source_binding_scope="all manifest CT/GT paths and file stats",
            validation_case_id=validation_id, validation_scope="one declared member; headers only",
            split_sha256_lf=split_hash, source_sha256_before=before)
        del sample, truth
        provenance = dict(fold=spec["fold"], split_fingerprint=split["manifest_fingerprint"],
            split_sha256_lf=split_hash, identity=identity, partitions=part,
            all_case_ids=[c.case_id for c in cases], all_group_ids=sorted({c.group_id for c in cases}),
            all_image_paths=sorted(str(c.image.resolve()) for c in cases),
            pilot_only=True, pilot_case_id=case.case_id, master_indices=masters,
            validation_subset=[validation_id], full_epoch_completed=False)

        def loader(selected):
            return DataLoader(dataset, batch_size=1, sampler=selected, num_workers=0,
                collate_fn=collate_slabs, generator=torch.Generator().manual_seed(config["training"]["seed"]))

        def forbidden_resize(*args, **kwargs):
            raise AssertionError("Native trial unexpectedly called XY interpolation")

        trainer = Trainer(config, loader(masters[:steps]), validation, provenance, scratch)
        if not getattr(trainer.model, "_activation_checkpointing_enabled", False):
            raise AssertionError("Production Trainer did not enable activation checkpointing")
        resolved = trainer.device
        report.update(trainer_class=type(trainer).__module__ + "." + type(trainer).__name__,
            activation_checkpointing_enabled=True,
            parameter_count=sum(p.numel() for p in trainer.model.parameters()),
            torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
            hardware=torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "CPU")
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
        initial = parameter_hash(trainer.model)
        with patch("merit_covid.data.pipeline.preprocessing.resize_in_plane", forbidden_resize):
            report["train"] = trainer.train_epoch(0)
        if report["train"]["optimizer_steps"] != steps or report["train"]["amp_skipped_steps"]:
            raise FloatingPointError("Not all requested initial optimizer updates completed")
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainer.model.parameters()):
            raise FloatingPointError("Missing/nonfinite production gradients")
        report["weights_changed"] = initial != parameter_hash(trainer.model)
        if not report["weights_changed"]:
            raise AssertionError("Production optimizer did not change learned parameters")
        expected_hash = state_hash(trainer.model)
        expected_steps = sorted(int(v["step"].item()) for v in trainer.optimizer.state.values() if "step" in v)
        with tempfile.TemporaryDirectory(prefix="main-pilot-", dir=scratch) as temporary:
            checkpoint = Path(temporary) / "diagnostic_only.pt"
            trainer.save_checkpoint(checkpoint, 0)
            report["checkpoint_bytes"] = checkpoint.stat().st_size
            del trainer
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
            resumed = Trainer(config, loader([masters[-1]]), validation, provenance, scratch)
            resumed.resume(checkpoint)
            actual_steps = sorted(int(v["step"].item()) for v in resumed.optimizer.state.values() if "step" in v)
            report["checkpoint"] = dict(model_exact_match=state_hash(resumed.model) == expected_hash,
                optimizer_step_counters_match=actual_steps == expected_steps and bool(expected_steps),
                start_epoch=resumed.start_epoch, diagnostic_only=True, full_epoch_completed=False)
            if (not report["checkpoint"]["model_exact_match"]
                    or not report["checkpoint"]["optimizer_step_counters_match"] or resumed.start_epoch != 1):
                raise AssertionError("Production checkpoint/resume mismatch")
            with patch("merit_covid.data.pipeline.preprocessing.resize_in_plane", forbidden_resize):
                report["after_resume"] = resumed.train_epoch(resumed.start_epoch)
            if report["after_resume"]["optimizer_steps"] != 1 or report["after_resume"]["amp_skipped_steps"]:
                raise FloatingPointError("No effective optimizer update after resume")
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in resumed.model.parameters()):
                raise FloatingPointError("Missing/nonfinite gradients after resume")
            model = resumed.model.eval()
            del resumed
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
            original_load = nib.load

            def ct_only_load(filename, *args, **kwargs):
                if Path(filename).resolve() == case.mask.resolve():
                    raise AssertionError("CT-only prediction opened GT")
                return original_load(filename, *args, **kwargs)

            with patch.object(nib, "load", ct_only_load), torch.inference_mode():
                ct_only = CTSlabDataset([case], preprocess, require_mask=False)
                batch = next(iter(DataLoader(ct_only, batch_size=1, sampler=[masters[0]], collate_fn=collate_slabs)))
                batch["image_slab"] = batch["image_slab"].to(resolved)
                logits = MERITUNETRPPAdapter(model).predict_logits(batch)
                if logits.shape != (1, 1, *config["model"]["img_size"][1:]) or not torch.isfinite(logits).all():
                    raise FloatingPointError("Invalid native master logits")
                probability = logits.float().sigmoid()
                report["inference"] = dict(ct_only_guard="PASS", master_only=True,
                    full_volume_reconstructed=False, shape=list(logits.shape),
                    probability_range=[float(probability.min()), float(probability.max())])
            if resolved.type == "cuda":
                report.update(peak_allocated_mib=torch.cuda.max_memory_allocated(resolved) / 2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved(resolved) / 2**20)
        report.update(status="PASS", checkpoint_deleted_after_verification=True)
    except Exception as error:
        report.update(status="OOM" if isinstance(error, torch.OutOfMemoryError) else "FAIL",
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if resolved is not None and resolved.type == "cuda" and torch.cuda.is_initialized():
            report.update(peak_allocated_mib=torch.cuda.max_memory_allocated(resolved) / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(resolved) / 2**20)
        if before is not None and case is not None:
            after = {"ct": file_hash(case.image), "gt": file_hash(case.mask)}
            report.update(source_sha256_after=after, source_files_unchanged=after == before)
            if after != before:
                report.update(status="FAIL", error="Source files changed")
        report["seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "spec", "output", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-cases", type=int, default=199)
    args = parser.parse_args()
    report = run_trial(args.config, args.manifest, json.loads(args.spec.read_text()),
                       args.output, args.scratch, steps=args.steps, device=args.device, expected_cases=args.expected_cases)
    print("Integrated main Trainer:", report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
