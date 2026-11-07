"""CLI giai đoạn dữ liệu: split, audit, smoke và evaluate.

Mỗi lệnh có một hàm run_* riêng. Muốn xem input cho train, dùng ví dụ
examples/read_training_batch.py thay vì tìm training loop trong file này.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ct_pipeline import CTSlabDataset, PreprocessConfig, make_dataloader
from ct_pipeline.cases import load_cases
from ct_pipeline.metrics import VolumeEvaluator
from ct_pipeline.preprocessing import read_volume
from ct_pipeline.splits import create_splits


def save_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def visualize(sample: dict, destination) -> None:
    """Xuất CT master, GT mask và overlay từ dữ liệu thật, không dùng model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    middle = sample["image_slab"].shape[1] // 2
    master = sample["image_slab"][0, middle].numpy()
    mask = sample["master_mask"][0].numpy()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].imshow(master, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[1].imshow(
        mask, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest"
    )
    axes[2].imshow(master, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[2].imshow(
        np.ma.masked_equal(mask, 0),
        cmap="autumn",
        vmin=0,
        vmax=1,
        alpha=0.45,
        origin="lower",
        interpolation="nearest",
    )
    for axis, title in zip(
        axes, ("Master CT", "Ground truth lesion", "CT + ground truth")
    ):
        axis.set_title(title)
        axis.set_xlabel("R ->")
        axis.set_ylabel("A ->")
        axis.set_xticks([])
        axis.set_yticks([])

    spacing = sample["metadata"]["native_spacing_mm"]
    figure.suptitle(
        f"{sample['case_id']} | master {sample['master_slice_index']} (0-based RAS)\n"
        f"native spacing XYZ = {spacing[0]:.3f}, {spacing[1]:.3f}, "
        f"{spacing[2]:.3f} mm; no Z resampling"
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Tập trung đường dẫn và tùy chọn CLI tại một chỗ."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("split", "audit", "smoke", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest", type=Path, default=ROOT / "data/covid19_20/cases.json"
        )
        command.add_argument("--expected-cases", type=int, default=199)
        command.add_argument(
            "--patient-map", type=Path, help="JSON mapping every case ID to a real patient ID"
        )

        if name != "split":
            command.add_argument(
                "--size", type=int, nargs=2, default=(256, 256), metavar=("H", "W")
            )
            command.add_argument("--slab-size", type=int, choices=(1, 3, 5, 11), default=11)
            command.add_argument("--hu-window", type=float, nargs=2, default=(-1000, 400))
            command.add_argument(
                "--intensity-encoding",
                choices=("auto_hu", "hu", "normalized_0_1"),
                default="auto_hu",
            )
            command.add_argument(
                "--assume-mm",
                action="store_true",
                help="Explicit override for documented mm data with unknown header units",
            )

        if name in ("smoke", "evaluate"):
            command.add_argument(
                "--split", type=Path, default=ROOT / "splits/covid19_20_5fold.json"
            )
            command.add_argument("--fold", type=int, choices=range(5), default=0)
            command.add_argument(
                "--partition",
                choices=("train", "validation", "test"),
                default="train" if name == "smoke" else "test",
            )

    split = commands.choices["split"]
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument(
        "--output", type=Path, default=ROOT / "splits/covid19_20_5fold.json"
    )
    commands.choices["audit"].add_argument(
        "--output", type=Path, default=ROOT / "analysis/covid19_20/audit.json"
    )

    smoke = commands.choices["smoke"]
    smoke.add_argument("--case-id")
    smoke.add_argument(
        "--master",
        type=int,
        help="0-based canonical index; default middle slice, independent of GT",
    )
    smoke.add_argument("--workers", type=int, default=0)
    smoke.add_argument(
        "--output", type=Path, default=ROOT / "outputs/covid19_20_slab.png"
    )
    smoke.add_argument(
        "--report", type=Path, default=ROOT / "analysis/covid19_20/smoke.json"
    )

    evaluate = commands.choices["evaluate"]
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory of case_id.npz containing indices and probabilities",
    )
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--hd95", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def preprocessing_from_args(args) -> PreprocessConfig:
    return PreprocessConfig(
        slab_size=args.slab_size,
        size_hw=tuple(args.size),
        hu_window=tuple(args.hu_window),
        intensity_encoding=args.intensity_encoding,
        assume_mm=args.assume_mm,
    )


def selected_dataset(args, config: PreprocessConfig) -> CTSlabDataset:
    return CTSlabDataset.from_split(
        manifest=args.manifest,
        split_path=args.split,
        fold=args.fold,
        partition=args.partition,
        config=config,
        patient_map=args.patient_map,
    )


def run_split(args, cases: list, identity: dict) -> None:
    """Lệnh split: tạo danh sách ID, không copy CT/mask hoặc chạy model."""
    result = create_splits(cases, identity, args.seed, args.validation_fraction)
    save_json(args.output, result)
    print(
        [
            {name: len(fold[name]["case_ids"]) for name in ("train", "validation", "test")}
            for fold in result["folds"]
        ]
    )
    print(args.output)


def run_audit(args, cases: list, identity: dict, config: PreprocessConfig) -> None:
    """Lệnh audit: đọc/kiểm tra từng cặp thật và lưu metadata hoặc lỗi."""
    valid, failures = [], []
    for index, case in enumerate(cases, start=1):
        try:
            image, mask, metadata = read_volume(case, config)
            valid.append({"case_id": case.case_id, "group_id": case.group_id, **metadata})
            del image, mask
            print(f"[{index}/{len(cases)}] OK {case.case_id}", flush=True)
        except (ValueError, OSError) as error:
            failures.append({"case_id": case.case_id, "error": str(error)})
            print(f"[{index}/{len(cases)}] FAIL {case.case_id}: {error}", flush=True)

    result = {
        "manifest_cases": len(cases),
        "valid_pairs": len(valid),
        "invalid_pairs": len(failures),
        "identity": identity,
        "config": asdict(config),
        "per_case": valid,
        "failures": failures,
        "z_spacing_mm_counts": dict(
            Counter(f"{item['native_spacing_mm'][2]:.6g}" for item in valid)
        ),
        "orientation_counts": dict(
            Counter("".join(item["original_orientation"]) for item in valid)
        ),
        "header_spatial_units": dict(
            Counter(item["spatial_unit_in_header"] for item in valid)
        ),
        "cases_using_assumed_mm": [
            item["case_id"] for item in valid
            if item["spatial_unit_in_header"] == "unknown"
        ],
        "intensity_note": (
            "auto_hu is a range sanity check after NIfTI scaling, "
            "not proof of DICOM HU provenance"
        ),
    }
    save_json(args.output, result)
    print(f"Valid {len(valid)}/{len(cases)}; invalid {len(failures)}; {args.output}")
    if failures:
        raise SystemExit(1)


def run_smoke(args, identity: dict, config: PreprocessConfig) -> None:
    """Lệnh smoke: lấy batch size 1/2 và vẽ một master, không train."""
    selected = selected_dataset(args, config)
    if args.case_id:
        case = next(
            (case for case in selected.cases if case.case_id == args.case_id), None
        )
    else:
        case = selected.cases[0]
    if case is None:
        raise ValueError("Requested case is not in the selected fold/partition")

    dataset = CTSlabDataset([case], config)
    master = len(dataset) // 2 if args.master is None else args.master
    sample = dataset[master]
    batches = {}
    for batch_size in (1, 2):
        loader = make_dataloader(
            dataset, batch_size=batch_size, num_workers=args.workers
        )
        batch = next(iter(loader))
        batches[str(batch_size)] = {
            "image_slab": list(batch["image_slab"].shape),
            "master_mask": list(batch["master_mask"].shape),
        }

    visualize(sample, args.output)
    result = {
        "device": "cpu",
        "case_id": case.case_id,
        "fold": args.fold,
        "partition": args.partition,
        "identity": identity,
        "config": asdict(config),
        "num_workers": args.workers,
        "image_slab_shape": list(sample["image_slab"].shape),
        "master_mask_shape": list(sample["master_mask"].shape),
        "master_slice_index": sample["master_slice_index"],
        "z_offsets_mm": sample["z_offsets_mm"].tolist(),
        "slice_indices": sample["slice_indices"].tolist(),
        "mask_values": torch.unique(sample["master_mask"]).tolist(),
        "batches": batches,
        "metadata": sample["metadata"],
        "visualization": str(args.output),
        "note": (
            "Real CT/ground truth data smoke test only; "
            "no model predictions or model metrics."
        ),
    }
    save_json(args.report, result)
    print(json.dumps(result, indent=2))


def run_evaluate(args, identity: dict, config: PreprocessConfig) -> None:
    """Lệnh evaluate: chỉ chạy khi caller đã có prediction cho từng ca trong split."""
    selected = selected_dataset(args, config)
    evaluator = VolumeEvaluator(config, args.threshold, args.hd95, identity)

    # Chỉ đánh giá cohort của partition đã chọn, không tự chọn ca theo thư mục output.
    for case in selected.cases:
        path = args.predictions / f"{case.case_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            indices = archive["indices"]
            predictions = archive["probabilities"]
            if (
                indices.ndim != 1
                or len(indices) != len(predictions)
                or not np.issubdtype(indices.dtype, np.integer)
            ):
                raise ValueError(f"Invalid indices/predictions in {path}")
            row = evaluator.evaluate_case(case, zip(indices, predictions))
            print(json.dumps(row), flush=True)

    report = evaluator.report()
    report.update(
        {"fold": args.fold, "partition": args.partition, "config": asdict(config)}
    )
    save_json(args.output, report)
    print(args.output)


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cases, identity = load_cases(args.manifest, args.expected_cases, args.patient_map)
    print(json.dumps(identity, ensure_ascii=False), flush=True)

    if args.command == "split":
        run_split(args, cases, identity)
        return

    config = preprocessing_from_args(args)
    if args.command == "audit":
        run_audit(args, cases, identity, config)
    elif args.command == "smoke":
        run_smoke(args, identity, config)
    else:
        run_evaluate(args, identity, config)


if __name__ == "__main__":
    main()
