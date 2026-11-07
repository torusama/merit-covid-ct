"""Xem dữ liệu mà code train sẽ nhận: chỉ đọc batch, không huấn luyện.

Từ thư mục repository:
    python examples/read_training_batch.py --assume-mm --partition train

Các đường dẫn mặc định lấy theo vị trí repository, không phụ thuộc ổ đĩa máy cũ.
Windows multiprocessing cần giữ lời gọi main() trong guard cuối file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ct_pipeline import PreprocessConfig, build_fold_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/covid19_20/cases.json"
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "splits/covid19_20_5fold.json"
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--partition", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--slab-size", type=int, choices=(1, 3, 5, 11), default=11)
    parser.add_argument("--assume-mm", action="store_true")
    parser.add_argument("--patient-map", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    # 1. Chọn cấu hình preprocessing. Muốn đổi input, bắt đầu từ đây.
    preprocess = PreprocessConfig(
        slab_size=args.slab_size,
        size_hw=tuple(args.size),
        hu_window=(-1000.0, 400.0),
        intensity_encoding="auto_hu",
        assume_mm=args.assume_mm,
    )

    # 2. Đọc fold ĐÃ LƯU. Ví dụ không shuffle để dễ đối chiếu các chỉ số.
    loaders = build_fold_loaders(
        manifest=args.manifest,
        split_path=args.split,
        fold=args.fold,
        preprocess=preprocess,
        batch_size=args.batch_size,
        shuffle_train=False,
        seed=args.seed,
        num_workers=args.workers,
        patient_map=args.patient_map,
    )
    selected_loader = getattr(loaders, args.partition)

    # 3. Đây là ba biến mà code model/train sau này sẽ đọc từ mỗi batch.
    batch = next(iter(selected_loader))
    inputs = batch["image_slab"]  # [B,1,D,H,W]
    targets = batch["master_mask"]  # [B,1,H,W]
    offsets_mm = batch["z_offsets_mm"]  # [B,D]

    # 4. Chỉ in hợp đồng đầu vào để kiểm tra; không tạo model/loss/optimizer.
    report = {
        "mode": "inspect_batch_only",
        "device": str(inputs.device),
        "fold": args.fold,
        "selected_partition": args.partition,
        "volumes_in_partition": {
            name: len(getattr(loaders, name).dataset.cases)
            for name in ("train", "validation", "test")
        },
        "master_samples_in_partition": {
            name: len(getattr(loaders, name).dataset)
            for name in ("train", "validation", "test")
        },
        "inputs": {"shape": list(inputs.shape), "dtype": str(inputs.dtype)},
        "targets": {
            "shape": list(targets.shape),
            "dtype": str(targets.dtype),
            "labels": torch.unique(targets).tolist(),
        },
        "z_offsets_mm_shape": list(offsets_mm.shape),
        "case_ids": batch["case_id"],
        "master_indices": batch["master_slice_index"].tolist(),
        "first_sample_offsets_mm": offsets_mm[0].tolist(),
        "first_sample_original_shape_xyz": batch["metadata"][0]["original_shape_xyz"],
        "note": "Patient-level independence remains unverified without patient IDs.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
