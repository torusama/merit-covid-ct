"""MERIT audit, train/resume, native evaluation, CV and synthetic software checks."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from merit_covid.engine.activation_checkpointing import configure_model_execution

from merit_covid.data.pipeline.cases import Case, load_cases
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.splits import validate_splits
from merit_covid.data.pipeline.metrics import aggregate_results
from merit_covid.data.manifest import load_manifest
from merit_covid.engine.training import (
    Trainer, load_experiment, split_datasets, evaluate_model, load_checkpoint_model, validate_experiment,
    load_development_split,
)

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def train_fold(config, fold, output, resume=None):
    datasets, provenance = split_datasets(config, fold)
    settings = config["training"]
    loader = make_dataloader(datasets["train"], batch_size=settings["batch_size"],
        shuffle=settings["shuffle_mode"] != "sequential", shuffle_mode=settings["shuffle_mode"],
        seed=settings["seed"], num_workers=settings["num_workers"])
    trainer = Trainer(config, loader, datasets["validation"].cases, provenance, output)
    history = trainer.fit(resume=resume)
    return trainer, history


def benchmark_step(config, device, count):
    """Function scope releases the previous graph/model before the next batch."""
    from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    model = configure_model_execution(MERITUNETRPP(**config["model"]).to(device), config["training"])
    criterion = MERITLoss(**config["loss"])
    image = torch.randn(count, 1, *model.img_size, device=device)
    masters = [model.img_size[0] // 2] * count
    offsets = (torch.arange(model.img_size[0], device=device)[None] - masters[0]).expand(count, -1).float().clone()
    offsets *= config["preprocess"].get("target_spacing_mm") or 1.0
    # Report model compute separately from initialization/native CUDA warm start.
    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_start = time.perf_counter()
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda" and config["training"]["amp"]):
        output = model(image, masters, offsets)
        losses = criterion(output, torch.zeros_like(image), masters, z_offsets_mm=offsets)
    losses["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"batch_size": count, "status": "completed", "seconds_forward_backward_cold": time.perf_counter()-compute_start,
        "seconds_including_initialization": time.perf_counter()-start,
        "finite_gradients": all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--headers-only", action="store_true")
    audit.add_argument("--limit", type=int)
    split = sub.add_parser("check-split")
    split.add_argument("--config", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=ROOT / "configs/merit_native512.yaml",
                       help="Experiment YAML; default: native512 with activation checkpointing")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--partition", choices=("validation", "test"), required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", type=Path, required=True)
    cv = sub.add_parser("cv")
    cv.add_argument("--config", type=Path, default=ROOT / "configs/merit_native512.yaml")
    cv.add_argument("--output", type=Path, required=True)
    external = sub.add_parser("external")
    external.add_argument("--checkpoint", type=Path, required=True)
    external.add_argument("--manifest", type=Path, required=True)
    external.add_argument("--confirm-frozen-protocol", action="store_true", required=True)
    external.add_argument("--device", default="cpu")
    external.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--output", type=Path, required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    bench.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    bench.add_argument("--output", type=Path, required=True)
    for command_parser in (audit, split, train, cv, bench):
        command_parser.add_argument("--target-spacing-mm", type=float,
            help="Explicit d for physical sampler; must be chosen on train/validation")
        command_parser.add_argument("--position-mode", choices=("index", "physical_mm"))
        command_parser.add_argument("--weighting-mode", choices=("index", "physical_mm"))
        command_parser.add_argument("--sigma-mm", type=float)
        command_parser.add_argument("--assume-mm", action="store_true", help="Explicit evidence-backed assumption for unknown NIfTI units")
    args = parser.parse_args()
    def experiment():
        return load_experiment(args.config, target_spacing_mm=args.target_spacing_mm,
            position_mode=args.position_mode, weighting_mode=args.weighting_mode,
            sigma_mm=args.sigma_mm, assume_mm=args.assume_mm)
    if args.command == "train":
        train_fold(experiment(), args.fold, args.output, args.resume)
    elif args.command in ("audit", "check-split"):
        config = experiment()
        cases, split_payload, identity, split_hash = load_development_split(config)
        provenance = {"split_sha256_lf": split_hash, "identity": identity,
                      "split_fingerprint": split_payload["manifest_fingerprint"],
                      "fold_counts": [{"fold": f["fold"], **{role: len(f[role]["case_ids"]) for role in ("train", "validation", "test")}}
                                      for f in split_payload["folds"]]}
        if args.command == "check-split":
            print(json.dumps(provenance, indent=2))
            return
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        rows = []
        for case in cases[:args.limit]:
            from merit_covid.data.pipeline.geometry import inspect_pair, affine_mm
            ct, mask = inspect_pair(case, PreprocessConfig(**config["preprocess"]))
            row = {"case_id": case.case_id, "group_id": case.group_id, "shape": list(ct.shape)}
            if not args.headers_only:
                _, labels, metadata = read_arrays(case, PreprocessConfig(**config["preprocess"]))
                # Native burden is descriptive audit only, never used to pick folds/samples.
                native = mask.get_fdata(dtype=np.float32, caching="unchanged")
                row.update(lesion_fraction=float(np.count_nonzero(native) / native.size),
                           metadata=metadata)
            rows.append(row)
        burden = {}
        if not args.headers_only and len(rows) == len(cases):
            by_id = {row["case_id"]: row["lesion_fraction"] for row in rows}
            for fold in split_payload["folds"]:
                burden[str(fold["fold"])] = {role: {"mean_fraction": float(np.mean([by_id[c] for c in fold[role]["case_ids"]])),
                                          "case_count": len(fold[role]["case_ids"])} for role in ("train", "validation", "test")}
        write_report(args.output, {"scope": "headers only" if args.headers_only else "native labels + preprocessing", "cases": rows,
                                  "fold_burden_descriptive_only": burden, "provenance": provenance})
    elif args.command == "evaluate":
        model, cp = load_checkpoint_model(args.checkpoint, args.device)
        datasets, provenance = split_datasets(cp["config"], cp["provenance"]["fold"])
        if provenance != cp["provenance"]:
            raise ValueError("Current development split differs from the frozen checkpoint")
        settings = cp["config"]["training"]
        report = evaluate_model(model, datasets[args.partition].cases, PreprocessConfig(**cp["config"]["preprocess"]),
            threshold=settings["threshold"], include_hd95=settings["include_hd95"],
            batch_size=settings["batch_size"], num_workers=settings["num_workers"], identity=provenance["identity"])
        write_report(args.output, {**report, "partition": args.partition, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                  "protocol_fingerprint": cp["protocol_fingerprint"], "provenance": provenance, "config": cp["config"]})
    elif args.command == "external":
        model, cp = load_checkpoint_model(args.checkpoint, args.device)
        records = load_manifest(args.manifest, dataset="zenodo20")
        if any(c.dataset != "zenodo20" for c in records) or len(records) != 20:
            raise ValueError("External evaluation requires exactly 20 Zenodo20 cases")
        if any(c.case_id in cp["provenance"]["all_case_ids"]
               or (c.patient_id or c.group_id) in cp["provenance"]["all_group_ids"]
               or str(c.image.resolve()) in cp["provenance"].get("all_image_paths", []) for c in records):
            raise ValueError("External/development identity overlap")
        cases = [Case(c.case_id, c.image, c.infection_mask, c.patient_id or c.group_id or c.case_id) for c in records]
        settings = cp["config"]["training"]
        report = evaluate_model(model, cases, PreprocessConfig(**cp["config"]["preprocess"]),
            threshold=settings["threshold"], include_hd95=settings["include_hd95"],
            batch_size=settings["batch_size"], identity={"patient_ids_available": all(c.patient_id for c in records) is True,
                                                      "identity_note": "Only supplied external linkage is known; cross-source identity is not independently verified"})
        write_report(args.output, {**report, "external": True, "protocol_fingerprint": cp["protocol_fingerprint"],
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "config": cp["config"]})
    elif args.command == "cv":
        config = experiment()
        reports = []
        for fold in range(5):
            directory = args.output / f"fold{fold}"
            train_fold(config, fold, directory)
            model, cp = load_checkpoint_model(directory / "best.pt", config["training"]["device"])
            datasets, provenance = split_datasets(config, fold)
            settings = config["training"]
            report = evaluate_model(model, datasets["test"].cases, PreprocessConfig(**config["preprocess"]),
                threshold=settings["threshold"], include_hd95=settings["include_hd95"], batch_size=settings["batch_size"],
                identity=provenance["identity"])
            report.update(fold=fold, protocol_fingerprint=cp["protocol_fingerprint"], provenance=provenance)
            write_report(directory / "test.json", report)
            reports.append(report)
        rows = [row for report in reports for row in report["per_case"]]
        summary = aggregate_results(rows, reports[0]["identity"])
        cross = {metric: {"mean": float(np.mean([r["macro"][metric]["mean"] for r in reports if r["macro"][metric]["mean"] is not None])),
                          "std": float(np.std([r["macro"][metric]["mean"] for r in reports if r["macro"][metric]["mean"] is not None]))}
                 for metric in ("dice", "iou", "precision", "recall")}
        write_report(args.output / "cv.json", {"folds": reports, "pooled_outer_test": summary, "fold_mean_std": cross})
    elif args.command == "smoke":
        from merit_covid.engine.synthetic import synthetic_experiment
        with TemporaryDirectory(prefix="merit-training-smoke-") as root:
            config = validate_experiment(synthetic_experiment(Path(root) / "data", device=args.device))
            trainer, first = train_fold(config, 0, Path(root) / "run")
            config["training"]["epochs"] = 2
            resumed, second = train_fold(config, 0, Path(root) / "run", Path(root) / "run" / "last.pt")
            model, cp = load_checkpoint_model(Path(root) / "run" / "best.pt", args.device)
            write_report(args.output, {"synthetic_only": True, "device": str(trainer.device),
                "first": first, "resumed": second, "best_epoch": cp["epoch"],
                "checkpoint_reload_verified": True, "training_completed": True,
                "real_ct_training": False})
    elif args.command == "benchmark":
        config = experiment()
        from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
        from merit_covid.engine.training import resolve_device
        device = resolve_device(args.device)
        torch.set_num_threads(config["training"]["threads"])
        rows = []
        for count in args.batch_sizes:
            if count < 1:
                raise ValueError("batch sizes must be positive")
            try:
                rows.append(benchmark_step(config, device, count))
            except torch.OutOfMemoryError:
                rows.append({"batch_size": count, "status": "out_of_memory"})
            print(json.dumps(rows[-1]), flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_report(args.output, {"synthetic_only": True, "torch": torch.__version__, "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "config": config, "measurements": rows, "timing_policy": "one cold forward/backward; not throughput benchmark"})


if __name__ == "__main__":
    main()
