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

from merit_covid.data.pipeline.cases import load_cases
from merit_covid import CTSlabDataset, PreprocessConfig
from merit_covid.data.pipeline.reporting import describe_case, summarize
from merit_covid.data.pipeline.splits import validate_splits


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
    window = source["hu_window"]
    config = PreprocessConfig(size_hw=None, hu_window=tuple(window) if window is not None else None,
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
        axes[row_index, 0].imshow(image, cmap="gray", origin="lower")
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
        axes[row_index, 2].imshow(image, cmap="gray", origin="lower")
        axes[row_index, 2].imshow(np.ma.masked_equal(mask, 0), cmap="autumn", vmin=0, vmax=1, alpha=0.45,
                                 origin="lower", interpolation="nearest")
        spacing = sample["metadata"]["native_spacing_mm"][2]
        axes[row_index, 0].set_ylabel(f"{case_id}\nmaster={master_index}, Z={spacing:.3g} mm", fontsize=9)
        axes[row_index, 0].set_title("Master CT")
        axes[row_index, 1].set_title("Ground-truth lesion")
        axes[row_index, 2].set_title("CT + ground truth")
        del dataset
    fig.suptitle("COVID-19-20 | middle slice examples, LPS orientation (L right, P up on plot)\n"
                 "Native H,W and tensor intensities; display auto-scaled per slice; no model inference")
    fig.savefig(output / "ct_mask_examples.png", dpi=130)
    plt.close(fig)


def write_report(summary, output):
    """Write the English primary report from saved numeric audit statistics."""
    assumed = ", ".join(f"`{c}`" for c in summary["cases_using_assumed_mm"])
    spacing_rows = "\n".join(
        f"| {float(k):.6g} | {v} |"
        for k, v in sorted(summary["spacing_z_mm_counts"].items(), key=lambda p: float(p[0]))
    )
    policy = summary.get("source_audit_config", {})
    report = f"""# COVID-19-20 dataset report

Generated: {summary['generated_at_bangkok']}. Descriptive CT/ground-truth evidence,
not model predictions.

## Processing and provenance

- Arrays are standardized to **LPS** by axis permutation/flips, not spatial interpolation.
- GT statistics use native resolution; no Z resampling is performed.
- Saved audit intensity policy: `{policy.get('intensity_encoding', 'unspecified')}`;
  HU window: `{json.dumps(policy.get('hu_window'))}`.
  Native mode preserves NIfTI-scaled values without clipping, normalization or HU inference.
- PNG grayscale is display-only; it does not change network input intensities.
- Canonical metadata refers to the LPS array grid. NiBabel world coordinates remain RAS+.
- Headers and full masks were rechecked against the saved audit. Full CT intensity/finite
  checks were reused from that audit; this report does not certify source byte identity.

## Artifacts

| File | Meaning |
| --- | --- |
| [case_statistics.csv](case_statistics.csv) | One row per CT volume; {summary['cases']} data rows |
| [summary.json](summary.json) | Cohort aggregates and per-case GT area trajectories |
| [dataset_overview.png](dataset_overview.png) | Slice spacing, volume length, lesion fraction and positive/empty slices |
| [lesion_trajectories.png](lesion_trajectories.png) | Native GT lesion area across physical stack distance |
| [ct_mask_examples.png](ct_mask_examples.png) | Middle-slice CT, GT and overlay, not AI predictions |

The complete 71-field schema and worked A-0025 example are in
[the dataset-report guide](../../docs/dataset_reports.md).

## Measured cohort statistics

- {summary['cases']} CT volumes; the number of distinct clinical patients is unverified.
- {summary['slices']:,} canonical slices: {summary['positive_slices']:,} with GT lesion,
  {summary['empty_slices']:,} with empty GT.
- Volume length: {summary['slice_count_min']} to {summary['slice_count_max']} slices;
  median {summary['slice_count_median']:g}.
- {summary['lesion_voxels']:,} lesion voxels / {summary['total_voxels']:,} total CT voxels.
- Pooled lesion / whole-CT fraction: {100 * summary['lesion_fraction_volume_pooled']:.4f}%.
- Mean per-volume lesion fraction: {100 * summary['lesion_fraction_volume_macro']:.4f}%.
  Pooled and macro summaries have different weighting and need not agree.
- Per-case native lesion counts agree with the saved audit.

| Original Z spacing (mm; assumptions apply where units are missing) | Volumes |
| --- | ---: |
{spacing_rows}

## Interpretation and limitations

- Current groups fall back to volume IDs. No verified patient-level independence.
  The report neither changes splits nor selects training cases.
- Unknown-unit cases explicitly assumed to be mm: {assumed or 'none'}.
  Their spacing, area and volume measurements depend on that assumption.
- No primary lung mask is available. Lesion fraction divides by the entire CT grid,
  including air outside the body; it is not percentage of infected lung.
- Adjacent GT Dice describes neighboring-label continuity, not model accuracy.
- This historical native-mask cohort report does not measure component loss under
  resizing. The newer runtime lesion-retention checker is documented separately.
- Saved scaled intensity range: {summary['ct_scaled_intensity_range']}.
  This is not proof of HU calibration.
- Examples use the first sorted case ID per spacing category and its middle slice,
  independently of GT; an example mask may be empty.
- Clinical linkage, intensity provenance and source byte identity still require verification.

## Regeneration

From the repository root, with the dataset and full audit already available:

~~~powershell
py tools/summarize_covid19_20.py --assume-mm
~~~

The flag deliberately accepts an unverified unit assumption. The command rechecks
headers/masks, reuses saved CT intensity checks and writes primary report artifacts.
CSV uses UTF-8 BOM for spreadsheet compatibility. Cases are processed one at a time.
No training, original voxel edits, external download or Zenodo20 analysis is performed.
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
