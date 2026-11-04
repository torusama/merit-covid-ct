"""Export primary-dataset CSV, summary and descriptive plots. Does not process Zenodo20."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ct_pipeline.cases import load_cases
from ct_pipeline.dataset import CTSlabDataset, PreprocessConfig
from ct_pipeline.reporting import describe_case, summarize
from ct_pipeline.splits import validate_splits


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def plot_overview(rows, summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    counts = sorted(summary["spacing_z_mm_counts"].items(), key=lambda p: float(p[0]))
    axes[0, 0].bar([f"{float(k):.3g}" for k, _ in counts], [v for _, v in counts], color="#3b82f6")
    axes[0, 0].set(xlabel="Native axial spacing (mm)", ylabel="Volumes", title="A. Distance between slices")
    for i, (_, count) in enumerate(counts):
        axes[0, 0].text(i, count, str(count), ha="center", va="bottom")
    axes[0, 1].hist([r["slices"] for r in rows], bins=20, color="#14b8a6", edgecolor="white")
    axes[0, 1].set(xlabel="Axial slices per volume", ylabel="Volumes", title="B. Volume lengths")
    axes[1, 0].hist([100 * r["lesion_fraction_volume"] for r in rows], bins=20, color="#f59e0b", edgecolor="white")
    axes[1, 0].set(xlabel="Lesion / complete CT volume (%)", ylabel="Volumes", title="C. Lesion burden (not % of lung)")
    order = np.arange(1, len(rows) + 1)
    positive = np.array([r["positive_slices"] for r in rows])
    axes[1, 1].bar(order, positive, color="#ef4444", width=1, label="With GT lesion")
    axes[1, 1].bar(order, [r["empty_slices"] for r in rows], bottom=positive, color="#cbd5e1", width=1, label="No GT lesion")
    axes[1, 1].set(xlabel="Volume order in case_statistics.csv", ylabel="Slices", title="D. Lesion-positive and empty slices")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"COVID-19-20 | {len(rows)} volumes | native data, no model predictions\n"
                 f"Units assumed mm for {len(summary['cases_using_assumed_mm'])} volumes with missing header units")
    fig.savefig(output / "dataset_overview.png", dpi=150)
    plt.close(fig)


def choose_examples(rows):
    # First case ID in each rounded spacing category; independent of lesion burden.
    chosen = {}
    for row in sorted(rows, key=lambda r: r["case_id"]):
        chosen.setdefault(round(row["slice_center_step_mm"], 3), row["case_id"])
    return [chosen[k] for k in sorted(chosen)]


def plot_trajectories(summary, example_ids, output):
    by_id = {t["case_id"]: t for t in summary["per_case_slice_areas"]}
    fig, axes = plt.subplots(len(example_ids), 1, figsize=(11, 2.8 * len(example_ids)), squeeze=False, constrained_layout=True)
    for ax, case_id in zip(axes[:, 0], example_ids):
        record = by_id[case_id]
        areas = np.asarray(record["areas_by_slice_voxels"]) * record["pixel_area_mm2"]
        distance = np.arange(len(areas)) * record["slice_center_step_mm"]
        ax.plot(distance, areas, color="#e11d48", linewidth=1.4)
        ax.fill_between(distance, areas, alpha=0.15, color="#e11d48")
        ax.set(title=f"{case_id} | slice step {record['slice_center_step_mm']:.3g} mm",
               xlabel="Distance from first canonical slice center (mm)", ylabel="GT lesion area (mm²)")
        ax.grid(alpha=0.2)
    fig.suptitle("Ground-truth lesion area across slices | examples selected by case ID within each spacing category\n"
                 "Descriptive mask statistics; not model predictions")
    fig.savefig(output / "lesion_trajectories.png", dpi=150)
    plt.close(fig)


def plot_ct_examples(cases, audit, example_ids, assume_mm, output):
    source = audit["config"]
    config = PreprocessConfig(size_hw=tuple(source["size_hw"]), hu_window=tuple(source["hu_window"]),
                              intensity_encoding=source["intensity_encoding"], assume_mm=assume_mm)
    by_id = {c.case_id: c for c in cases}
    fig, axes = plt.subplots(len(example_ids), 3, figsize=(12, 3.8 * len(example_ids)), squeeze=False, constrained_layout=True)
    for row_index, case_id in enumerate(example_ids):
        dataset = CTSlabDataset([by_id[case_id]], config)
        master_index = len(dataset) // 2  # Independent of GT lesion location.
        sample = dataset[master_index]
        image = sample["image_slab"][0, config.slab_size // 2].numpy()
        mask = sample["master_mask"][0].numpy()
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
        axes[row_index, 2].imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[row_index, 2].imshow(np.ma.masked_equal(mask, 0), cmap="autumn", vmin=0, vmax=1, alpha=0.45,
                                 origin="lower", interpolation="nearest")
        spacing = sample["metadata"]["native_spacing_mm"][2]
        axes[row_index, 0].set_ylabel(f"{case_id}\nmaster={master_index}, Z={spacing:.3g} mm", fontsize=9)
        axes[row_index, 0].set_title("Master CT")
        axes[row_index, 1].set_title("Ground-truth lesion")
        axes[row_index, 2].set_title("CT + ground truth")
        del dataset
    fig.suptitle("COVID-19-20 | middle slice examples, RAS orientation (R right, A up)\n"
                 "In-plane preprocessing only; no Z resampling or model inference")
    fig.savefig(output / "ct_mask_examples.png", dpi=130)
    plt.close(fig)


def write_report(summary, output):
    assumed = ", ".join(f"`{c}`" for c in summary["cases_using_assumed_mm"])
    spacing_rows = "\n".join(f"| {float(k):.6g} | {v} |" for k, v in sorted(summary["spacing_z_mm_counts"].items(), key=lambda p: float(p[0])))
    report = f"""# Tổng quan dữ liệu COVID-19-20

Ngày tạo: {summary['generated_at_bangkok']}. Báo cáo mô tả dữ liệu thật; chưa có model prediction.

## Nên mở file nào?

| File | Cách đọc |
| --- | --- |
| [case_statistics.csv](case_statistics.csv) | Mở bằng Excel: **mỗi dòng là một CT volume**, tổng {summary['cases']} dòng dữ liệu. |
| [summary.json](summary.json) | Các con số tổng hợp và số voxel tổn thương trên từng lát của từng volume. |
| [dataset_overview.png](dataset_overview.png) | Bốn biểu đồ: spacing Z, số lát mỗi volume, tỷ lệ tổn thương và số lát có/không có tổn thương. |
| [lesion_trajectories.png](lesion_trajectories.png) | Diện tích GT tổn thương thay đổi qua các lát của các volume ví dụ. |
| [ct_mask_examples.png](ct_mask_examples.png) | CT, GT mask và ảnh chồng mask lên CT; lấy lát giữa volume, không chọn lát theo GT. |

## Các con số chính

- {summary['cases']} CT volumes; chưa xác minh tương ứng với bao nhiêu bệnh nhân khác nhau.
- {summary['slices']:,} lát canonical: {summary['positive_slices']:,} lát có ít nhất một voxel tổn thương,
  {summary['empty_slices']:,} lát có mask rỗng.
- Mỗi volume có từ {summary['slice_count_min']} đến {summary['slice_count_max']} lát;
  median là {summary['slice_count_median']:g} lát.
- {summary['lesion_voxels']:,} voxel tổn thương trên tổng {summary['total_voxels']:,} voxel CT.
- Tỷ lệ gộp lesion/toàn bộ CT: {100 * summary['lesion_fraction_volume_pooled']:.4f}%.
  Trung bình tỷ lệ riêng từng volume: {100 * summary['lesion_fraction_volume_macro']:.4f}%.
  Hai cách tổng hợp có trọng số khác nhau, nên có thể cho kết quả khác nhau.
- Tất cả headers CT/mask được kiểm tra lại geometry; toàn bộ mask được đọc lại,
  kiểm tra finite và nhãn 0/1. Tổng voxel tổn thương từng ca khớp audit cũ.

| Spacing Z gốc (mm, có giả định cho các ca thiếu đơn vị) | Số volume |
| --- | ---: |
{spacing_rows}

## Các cột CSV thường dùng

| Cột | Giải thích |
| --- | --- |
| `volume_order`, `case_id` | Số thứ tự tương ứng biểu đồ và mã volume. |
| `group_id`, `patient_id`, `patient_ids_available` | Nhóm dùng chia dữ liệu; patient ID để trống khi chưa biết. Group hiện là volume ID. |
| `image`, `mask` | Đường dẫn file CT và mask theo thư mục chứa manifest. |
| `original_shape_x/y/z` | Kích thước NIfTI gốc; trục thứ ba gốc có thể bị đảo/hoán vị khi reorient. |
| `slices` | Số lát axial sau reorient RAS, trước mọi resize/padding. |
| `spacing_x/y/z_mm`, `original_orientation` | Khoảng cách voxel và hướng của ảnh gốc. |
| `spatial_unit_in_header`, `spatial_units_assumed_mm` | Header có ghi đơn vị không, và ca đó có phải giả định mm không. |
| `ct_min`, `ct_max`, `ct_dtype`, `ct_scaling_slope/intercept` | Thông tin intensity CT. Min/max lấy lại từ audit đầy đủ trước đây; dtype/scaling đọc header hiện tại. |
| `positive_slices`, `empty_slices` | Số lát có và không có tổn thương GT. |
| `lesion_voxels`, `total_voxels` | Số voxel tổn thương và số voxel của toàn volume. |
| `lesion_fraction_volume` | Tỷ lệ 0–1 của lesion trên toàn bộ CT, **không phải tỷ lệ trên phổi**. |
| `lesion_volume_ml` | Thể tích GT tổn thương, đơn vị mL, dựa trên determinant của affine. |
| `first/last_positive_slice` | Chỉ số canonical 0-based đầu/cuối có tổn thương; trống nếu không có. |
| `positive_run_count/lengths`, `max_positive_run_length` | Số đoạn lát liên tiếp có tổn thương và độ dài các đoạn. |
| `adjacent_positive_pairs`, `adjacent_gt_dice_median` | Số cặp GT liên tiếp cùng có tổn thương và median Dice giữa các GT đó; **không phải Dice model**. |
| `fold_0` đến `fold_4` | Ca đó thuộc train, validation hay test trong từng fold của split đã lưu. |
| `geometry_match`, `mask_finite_binary`, `z_resampled` | Kết quả kiểm tra geometry/mask và việc có nội suy Z hay không. |
| Các cột `audit_*` | Giữ toàn bộ trường của audit gốc; list/matrix được lưu dạng JSON trong một ô CSV. |

## Giới hạn cần đọc cùng số liệu

- **Patient-level independence chưa xác minh**: chưa biết hai volume khác nhau có cùng người hay không.
  Không suy diễn patient ID từ tên file. Báo cáo không thay đổi split hoặc chọn dữ liệu training.
- Các ca sau thiếu đơn vị trong header, đang chủ động giả định mm: {assumed or 'không có'}.
  Spacing, diện tích và thể tích của chúng phụ thuộc giả định này.
- Không có lung mask nên không tính phần trăm phổi bị tổn thương hoặc tổn thương ngoài phổi.
  `lesion_fraction_volume` lấy mẫu số là toàn bộ khối CT, gồm cả vùng không khí bên ngoài cơ thể.
- Không tính connected components hoặc mất thành phần do resize trong bản báo cáo này;
  các chỉ số đó có ở báo cáo Zenodo20. Báo cáo hiện dùng độ phân giải mask gốc.
- Không đọc lại toàn bộ intensity CT của 199 ca: dùng min/max và kết quả kiểm tra intensity từ
  audit cũ; headers và mask được kiểm tra lại. Không có checksum để xác minh byte identity CT.
- Intensity gộp trong audit: {summary['ct_scaled_intensity_range']}. Đây không phải chứng nhận calibration HU.
- Ví dụ CT được chọn theo case ID đầu tiên trong từng nhóm spacing, rồi lấy lát giữa;
  mask ở lát minh họa có thể rỗng. Không dùng GT để chọn ROI đầu vào.

## Chạy lại bằng PowerShell

Chạy từ thư mục gốc repository khi đã có dataset và audit:

```powershell
.\\.venv\\Scripts\\python.exe tools/summarize_covid19_20.py --assume-mm
```

CSV dùng UTF-8 BOM để hỗ trợ Excel. Tổng hợp từng mask một; không nạp cả cohort vào RAM.
Script không train model, không sửa voxel gốc, không tải hoặc xử lý thêm Zenodo20.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/covid19_20/cases.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "analysis/covid19_20/audit.json")
    parser.add_argument("--split", type=Path, default=ROOT / "splits/covid19_20_5fold.json")
    parser.add_argument("--patient-map", type=Path)
    parser.add_argument("--expected-cases", type=int, default=199)
    parser.add_argument("--assume-mm", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/covid19_20")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cases, identity = load_cases(args.manifest, args.expected_cases, args.patient_map)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    records = {r["case_id"]: r for r in audit["per_case"]}
    if len(records) != len(audit["per_case"]) or set(records) != {c.case_id for c in cases} or audit["invalid_pairs"]:
        raise ValueError("Need one successful audit record for exactly every manifest case")
    if audit["identity"] != identity:
        raise ValueError("Audit identity provenance differs from current manifest/patient mapping")
    manifest_rows = {r["id"]: r for r in json.loads(args.manifest.read_text(encoding="utf-8-sig"))}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    validate_splits(split, cases)
    membership = {c.case_id: {} for c in cases}
    for fold in split["folds"]:
        for partition in ("train", "validation", "test"):
            for case_id in fold[partition]["case_ids"]:
                membership[case_id][f"fold_{fold['fold']}"] = partition
    rows, trajectories = [], []
    for index, case in enumerate(cases, start=1):
        row, trajectory = describe_case(case, records[case.case_id], args.assume_mm)
        rows.append({"volume_order": index, "case_id": row.pop("case_id"),
                     "patient_id": case.group_id if identity["patient_ids_available"] else "",
                     "patient_ids_available": identity["patient_ids_available"],
                     "image": manifest_rows[case.case_id]["image"], "mask": manifest_rows[case.case_id]["mask"],
                     **row, **membership[case.case_id]})
        trajectories.append(trajectory)
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
    summary = summarize(rows, trajectories, identity, audit["config"])
    summary.update({"generated_at_bangkok": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds"),
                    "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest()})
    examples = choose_examples(rows)
    summary["example_case_ids"] = examples
    summary["example_selection"] = "First sorted case ID in each rounded slice-spacing category; middle axial slice, independent of GT"
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "case_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "summary.json", summary)
    plot_overview(rows, summary, args.output)
    plot_trajectories(summary, examples, args.output)
    plot_ct_examples(cases, audit, examples, args.assume_mm, args.output)
    write_report(summary, args.output)
    print(f"Wrote {len(rows)} CSV rows, summary, 3 PNGs and REPORT.md to {args.output}")


if __name__ == "__main__":
    main()
