"""Minimal MERIT training/validation with epoch-boundary reproducible resume.

Uses the ONE shared slab Dataset/reader and native-volume evaluator. No dataset
download, split generation, test tuning, or implicit external evaluation.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
import platform
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
import yaml

from merit_covid.data.pipeline.cases import load_cases
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.data.pipeline.geometry import inspect_pair
from merit_covid.data.pipeline.metrics import aggregate_results, volume_metrics
from merit_covid.data.pipeline.splits import validate_splits
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
from .slab_inference import predict_slab_case


TRAIN_DEFAULTS = dict(epochs=100, batch_size=1, num_workers=0, learning_rate=3e-4,
    weight_decay=1e-4, device="auto", amp=True, amp_init_scale=1024.0, gradient_clip=1.0,
    seed=42, warmup_epochs=30, patience=20, scheduler_patience=5, scheduler_factor=0.5,
    threshold=0.5, include_hd95=True, threads=2, deterministic=True)


def json_ready(value):
    """Normalize YAML lists / dataclass tuples for stable checkpoint comparisons."""
    return json.loads(json.dumps(value, allow_nan=False))


def explicit_parameters(cls, supplied):
    """Persist defaults too; a future source default must not silently alter a run."""
    parameters = inspect.signature(cls.__init__).parameters
    defaults = {name: p.default for name, p in parameters.items()
                if name != "self" and p.default is not inspect.Parameter.empty}
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {unknown}")
    return {**defaults, **supplied}


def load_experiment(path, *, target_spacing_mm=None, position_mode=None, weighting_mode=None,
                    sigma_mm=None, assume_mm=False):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"data", "preprocess", "model", "loss", "training"}:
        raise ValueError("Experiment requires data/preprocess/model/loss/training mappings")
    for key in ("manifest", "split_path", "patient_map"):
        if config["data"].get(key) is not None:
            config["data"][key] = str((path.parent / config["data"][key]).resolve())
    if target_spacing_mm is not None:
        config["preprocess"]["target_spacing_mm"] = target_spacing_mm
    if position_mode is not None:
        config["model"]["position_mode"] = position_mode
    if weighting_mode is not None:
        config["loss"]["weighting_mode"] = weighting_mode
    if sigma_mm is not None:
        config["loss"]["sigma_mm"] = sigma_mm
    if assume_mm:
        config["preprocess"]["assume_mm"] = True
    return validate_experiment(config)


def validate_experiment(config):
    config = deepcopy(config)
    if set(config) != {"data", "preprocess", "model", "loss", "training"}:
        raise ValueError("Unexpected experiment fields")
    if not all(isinstance(config[k], dict) for k in config):
        raise ValueError("Experiment sections must be mappings")
    if set(config["data"]) - {"manifest", "split_path", "patient_map"} or not {"manifest", "split_path"} <= set(config["data"]):
        raise ValueError("data needs manifest/split_path and optional patient_map")
    unknown = set(config["training"]) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training settings: {unknown}")
    settings = {**TRAIN_DEFAULTS, **config["training"]}
    for key in ("epochs", "batch_size", "threads", "warmup_epochs", "patience", "scheduler_patience"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    for key in ("learning_rate", "gradient_clip", "amp_init_scale"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(settings["weight_decay"]) or settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    if not 0 < settings["scheduler_factor"] < 1 or not 0 <= settings["threshold"] <= 1:
        raise ValueError("Invalid scheduler_factor/threshold")
    for key in ("amp", "deterministic", "include_hd95"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    config["training"] = settings
    config["model"] = explicit_parameters(MERITUNETRPP, config["model"])
    config["loss"] = explicit_parameters(MERITLoss, config["loss"])
    preprocess = PreprocessConfig(**config["preprocess"])
    config["preprocess"] = asdict(preprocess)
    grid = tuple(config["model"].get("img_size", (11, 128, 128)))
    if preprocess.slab_size != grid[0] or tuple(preprocess.size_hw or ()) != grid[1:]:
        raise ValueError("Training requires explicit size_hw/slab_size matching model img_size")
    # Sampling, PE and Gaussian weighting are deliberately independent ablation axes.
    criterion = MERITLoss(**config["loss"])
    config["loss"]["gaussian_sigma"] = criterion.gaussian_sigma
    config["loss"]["sigma_slices"] = criterion.gaussian_sigma
    return json_ready(config)


def load_development_split(config):
    data = config["data"]
    cases, identity = load_cases(data["manifest"], patient_map=data.get("patient_map"))
    raw = Path(data["split_path"]).read_bytes()
    split = json.loads(raw.decode("utf-8-sig"))
    validate_splits(split, cases)
    return cases, split, identity, hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def split_datasets(config, fold):
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be 0..4")
    cases, split, identity, split_hash = load_development_split(config)
    part = next(f for f in split["folds"] if f["fold"] == fold)
    datasets = {}
    for role in ("train", "validation", "test"):
        ids = set(part[role]["case_ids"])
        datasets[role] = CTSlabDataset([c for c in cases if c.case_id in ids],
                                     PreprocessConfig(**config["preprocess"]))
        datasets[role].partition = role
    provenance = {"fold": fold, "split_fingerprint": split["manifest_fingerprint"],
        "split_sha256_lf": split_hash,
        "identity": identity, "partitions": part,
        "all_case_ids": [c.case_id for c in cases], "all_group_ids": sorted({c.group_id for c in cases})}
    provenance["all_image_paths"] = sorted(str(c.image.resolve()) for c in cases)
    return datasets, provenance


def protocol_fingerprint(config, provenance):
    payload = deepcopy(config)
    payload["training"].pop("epochs")  # Extending the epoch budget is allowed on resume.
    payload["data"] = provenance  # Freeze split identity and resolved source-path provenance.
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def evaluate_model(model, cases, preprocess, *, threshold=0.5, include_hd95=True,
                   batch_size=1, num_workers=0, identity=None):
    """Original CT grid metrics; masks read only AFTER CT-only prediction."""
    rows = []
    was_training = model.training
    model.eval()
    try:
        for case in cases:
            inspect_pair(case, preprocess, require_mask=True)
            prediction = predict_slab_case(model, case, preprocess, batch_size=batch_size, num_workers=num_workers)
            truth = nib.load(str(case.mask)).get_fdata(dtype=np.float32, caching="unchanged")
            native = prediction.to_original_xyz()
            metrics = volume_metrics(native, truth, threshold,
                prediction.metadata["original_affine_mm"], include_hd95)
            rows.append({"case_id": case.case_id, "group_id": case.group_id, **metrics,
                         "lesion_retention": _retention_for_case(case, preprocess)})
    finally:
        model.train(was_training)
    return {**aggregate_results(rows, identity), "threshold": threshold,
            "threshold_policy": "fixed a priori in frozen experiment config, not test-tuned",
            "evaluation_grid": "full original NIfTI XYZ", "preprocess": asdict(preprocess)}


def _retention_for_case(case, preprocess):
    # Already reported by training Dataset; evaluation labels stay on native grid.
    # Report resized-GT loss without reading CT a second time.
    from merit_covid.data.pipeline.geometry import reorient_to_lps
    from merit_covid.data.pipeline.preprocessing import lesion_retention
    mask = reorient_to_lps(nib.load(str(case.mask))).get_fdata(dtype=np.float32, caching="unchanged")
    return lesion_retention(mask.transpose(2, 1, 0), preprocess.size_hw)


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Training requires an available CPU or CUDA device")
    return device


class Trainer:
    def __init__(self, config, train_loader, validation_cases, provenance, output_dir):
        self.config = validate_experiment(config)
        self.settings = self.config["training"]
        self.provenance = deepcopy(provenance)
        self.fingerprint = protocol_fingerprint(self.config, provenance)
        self.output_dir = Path(output_dir)
        self.train_loader, self.validation_cases = train_loader, list(validation_cases)
        if not self.validation_cases or not len(train_loader):
            raise ValueError("Training and validation must be nonempty")
        if not getattr(train_loader.dataset, "require_mask", False):
            raise ValueError("Full-slab training requires true GT")
        if json_ready(asdict(train_loader.dataset.config)) != self.config["preprocess"]:
            raise ValueError("Training Dataset preprocessing differs from frozen experiment config")
        train_cases = train_loader.dataset.cases
        if ({c.case_id for c in train_cases} & {c.case_id for c in self.validation_cases}
                or {c.group_id for c in train_cases} & {c.group_id for c in self.validation_cases}):
            raise ValueError("Train/validation case or group leakage")
        seed = self.settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(self.settings["threads"])
        torch.use_deterministic_algorithms(self.settings["deterministic"])
        torch.backends.cudnn.benchmark = False
        self.device = resolve_device(self.settings["device"])
        self.model = MERITUNETRPP(**self.config["model"]).to(self.device)
        self.criterion = MERITLoss(**self.config["loss"])
        self.preprocess = PreprocessConfig(**self.config["preprocess"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.settings["learning_rate"],
                                           weight_decay=self.settings["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max",
            factor=self.settings["scheduler_factor"], patience=self.settings["scheduler_patience"])
        self.amp_enabled = self.settings["amp"] and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled, init_scale=self.settings["amp_init_scale"])
        self.start_epoch, self.best_metric, self.bad_epochs = 0, -math.inf, 0

    def train_epoch(self, epoch):
        self.model.train()
        self.model.set_epoch(epoch, warmup_epochs=self.settings["warmup_epochs"])
        totals, examples, steps, skipped = {}, 0, 0, 0
        for index, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            image = batch["image_slab"].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                output = self.model(image, batch["master_index"], batch["z_offsets_mm"])
                losses = self.criterion(output, batch["mask_slab"].to(self.device), batch["master_index"],
                    valid_region=batch["valid_region"].to(self.device), z_offsets_mm=batch["z_offsets_mm"])
            if not all(torch.isfinite(v).all() for v in losses.values()):
                raise FloatingPointError("Non-finite loss; no optimizer step performed")
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings["gradient_clip"],
                                           error_if_nonfinite=not self.amp_enabled)
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            overflow = self.scaler.get_scale() < previous_scale
            skipped += int(overflow)
            steps += int(not overflow)
            examples += len(image)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(image)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(self.train_loader):
                print(f"epoch={epoch} batch={index+1}/{len(self.train_loader)} loss={float(losses['total'].detach()):.6f} ramp={float(self.model.pcg_ramp):.3f}", flush=True)
        return {**{key: value / examples for key, value in totals.items()},
                "optimizer_steps": steps, "amp_skipped_steps": skipped, "examples": examples}

    def validate(self):
        return evaluate_model(self.model, self.validation_cases, self.preprocess,
            threshold=self.settings["threshold"], include_hd95=self.settings["include_hd95"],
            batch_size=self.settings["batch_size"], num_workers=self.settings["num_workers"],
            identity=self.provenance.get("identity"))

    def save_checkpoint(self, path, epoch):
        payload = {"schema_version": 1, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(), "epoch": epoch,
            "best_metric": self.best_metric, "bad_epochs": self.bad_epochs, "config": self.config,
            "provenance": self.provenance, "protocol_fingerprint": self.fingerprint,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None},
            "loader_generator": self.train_loader.generator.get_state() if self.train_loader.generator is not None else None}
        payload["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "resolved_device": str(self.device), "device_type": self.device.type,
            "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.processor(),
            "cuda_runtime": torch.version.cuda}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def resume(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != 1 or checkpoint.get("protocol_fingerprint") != self.fingerprint:
            raise ValueError("Checkpoint config/preprocessing/sampling/split/fold mismatch")
        if checkpoint.get("environment", {}).get("device_type", self.device.type) != self.device.type:
            raise ValueError("Resume device type changed; exact RNG replay across CPU/CUDA is not guaranteed")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric, self.bad_epochs = checkpoint["best_metric"], checkpoint["bad_epochs"]
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if self.device.type == "cuda" and checkpoint["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        if self.train_loader.generator is not None and checkpoint["loader_generator"] is not None:
            self.train_loader.generator.set_state(checkpoint["loader_generator"])

    def fit(self, *, resume=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume is None and (self.output_dir / "last.pt").exists():
            raise FileExistsError("Existing run found; choose a new output directory or --resume")
        if resume is not None:
            self.resume(resume)
        history = []
        for epoch in range(self.start_epoch, self.settings["epochs"]):
            if self.bad_epochs >= self.settings["patience"]:
                break
            train = self.train_epoch(epoch)
            validation = self.validate()
            score = validation["macro"]["dice"]["mean"]
            if score is None or not math.isfinite(score):
                raise FloatingPointError("Validation Dice undefined/non-finite")
            improved = score > self.best_metric
            self.best_metric = max(score, self.best_metric)
            self.bad_epochs = 0 if improved else self.bad_epochs + 1
            self.scheduler.step(score)
            row = {"epoch": epoch, "train": train, "validation": validation, "best_dice": self.best_metric,
                   "pcg_ramp": float(self.model.pcg_ramp), "learning_rate": self.optimizer.param_groups[0]["lr"]}
            if improved:
                self.save_checkpoint(self.output_dir / "best.pt", epoch)
            self.save_checkpoint(self.output_dir / "last.pt", epoch)
            with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            history.append(row)
            print(f"epoch={epoch} validation_native_macro_dice={score:.6f} best={self.best_metric:.6f}", flush=True)
        return history


def load_checkpoint_model(path, device="cpu"):
    """Trusted local checkpoints only; no pickle fetched from external services."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if protocol_fingerprint(checkpoint["config"], checkpoint["provenance"]) != checkpoint["protocol_fingerprint"]:
        raise ValueError("Checkpoint protocol metadata is inconsistent")
    torch.set_num_threads(checkpoint["config"]["training"]["threads"])
    model = MERITUNETRPP(**checkpoint["config"]["model"]).to(resolve_device(device))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
