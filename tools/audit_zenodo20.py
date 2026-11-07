"""Reproducible audit of the local 20-case COVID-19 CT segmentation dataset."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from merit_covid.data.manifest import legacy_mask_path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo20"
OUT = ROOT / "analysis" / "zenodo20"


def percentile(values, points=(0, 10, 25, 50, 75, 90, 100)):
    return {str(p): float(np.percentile(values, p)) for p in points} if len(values) else {}


def true_runs(flags):
    runs, length = [], 0
    for flag in flags:
        if flag:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def main():
    cases = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    rows = []
    all_areas = []
    all_components = []
    adjacent_dice = []
    all_runs = []
    intensity_by_source = {"Coronacases": [], "Radiopaedia": []}
    label_counts = Counter()
    total_lost_components = {128: 0, 256: 0, 352: 0}
    total_components = 0
    total_lost_positive_slices = {128: 0, 256: 0, 352: 0}
    total_gt_positive_slices = 0
    infection_encoding_equal = True
    combined_union_equal = True
    geometry_equal = True
    per_case = []

    for case in cases:
        identifier = case["id"]
        print(identifier, flush=True)
        image = nib.load(str(DATA / case["image"]))
        combined = nib.load(str(DATA / case["combined_mask"]))
        infection = nib.load(str(legacy_mask_path(case, DATA, "mask")))
        lung = nib.load(str(legacy_mask_path(case, DATA, "lung_mask")))
        volumes = (image, combined, infection, lung)
        same_geometry = all(
            x.shape == image.shape
            and np.allclose(x.affine, image.affine)
            and np.allclose(x.header.get_zooms()[:3], image.header.get_zooms()[:3])
            for x in volumes[1:]
        )
        geometry_equal &= same_geometry

        # The original Zenodo masks are gzip NIfTI inside ZIP files.  Decode one
        # case at a time; the CT and Kaggle combined mask remain memory-mapped.
        inf = np.asarray(infection.dataobj)
        lung_data = np.asarray(lung.dataobj)
        infection_binary = inf > 0
        lung_binary = lung_data > 0
        case_counts = Counter()
        case_areas = []
        case_component_sizes = []
        case_adjacent_dice = []
        positive_flags = []
        multi_component_slices = 0
        large_area_change_pairs = 0
        source_sample = np.asarray(image.dataobj[::8, ::8, ::4], dtype=np.float32)
        intensity_by_source[case["source"]].append(source_sample.ravel())
        prev_mask = None
        prev_area = None
        infection_match = True
        union_match = True

        for z in range(image.shape[2]):
            lesion = infection_binary[:, :, z]
            lung_slice = lung_binary[:, :, z]
            combined_slice = np.asarray(combined.dataobj[:, :, z])
            infection_match &= np.array_equal(lesion, combined_slice == 3)
            union_match &= np.array_equal(lung_slice | lesion, combined_slice > 0)
            labels, counts = np.unique(combined_slice, return_counts=True)
            case_counts.update({int(k): int(v) for k, v in zip(labels, counts)})

            area = int(lesion.sum())
            case_areas.append(area)
            positive_flags.append(area > 0)
            if area:
                total_gt_positive_slices += 1
                binary = lesion.astype(np.uint8)
                count, component_map, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
                sizes = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
                case_component_sizes.extend(sizes)
                total_components += len(sizes)
                multi_component_slices += len(sizes) > 1
                for resolution in (128, 256, 352):
                    shrunk = cv2.resize(component_map, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
                    survived = np.unique(shrunk)
                    survived = survived[survived > 0]
                    total_lost_components[resolution] += len(sizes) - len(survived)
                    total_lost_positive_slices[resolution] += len(survived) == 0

            if prev_mask is not None and area and prev_area:
                intersection = int(np.logical_and(lesion, prev_mask).sum())
                score = 2 * intersection / (area + prev_area)
                case_adjacent_dice.append(score)
                if max(area, prev_area) / min(area, prev_area) > 2:
                    large_area_change_pairs += 1
            prev_mask = lesion
            prev_area = area

        infection_encoding_equal &= infection_match
        combined_union_equal &= union_match
        label_counts.update(case_counts)
        runs = true_runs(positive_flags)
        all_runs.extend(runs)
        all_components.extend(case_component_sizes)
        adjacent_dice.extend(case_adjacent_dice)
        all_areas.extend(area for area in case_areas if area)
        xy = image.header.get_zooms()[:3]
        total_voxels = int(np.prod(image.shape))
        infected_voxels = int(infection_binary.sum())
        lung_voxels = int(lung_binary.sum())
        outside_lung = int(np.logical_and(infection_binary, ~lung_binary).sum())
        row = {
            "id": identifier, "source": case["source"],
            "shape": "x".join(map(str, image.shape)),
            "slices": image.shape[2],
            "spacing_x_mm": float(xy[0]), "spacing_y_mm": float(xy[1]), "spacing_z_mm": float(xy[2]),
            "orientation": "".join(nib.aff2axcodes(image.affine)),
            "ct_dtype": str(image.get_data_dtype()),
            "ct_sample_p01": float(np.percentile(source_sample, 1)),
            "ct_sample_p50": float(np.percentile(source_sample, 50)),
            "ct_sample_p99": float(np.percentile(source_sample, 99)),
            "positive_slices": sum(positive_flags),
            "empty_slices": len(positive_flags) - sum(positive_flags),
            "infection_voxels": infected_voxels,
            "lung_voxels": lung_voxels,
            "all_voxels": total_voxels,
            "infection_fraction_all": infected_voxels / total_voxels,
            "infection_fraction_lung": infected_voxels / lung_voxels if lung_voxels else None,
            "infection_outside_lung": outside_lung,
            "components_2d": len(case_component_sizes),
            "multicomponent_positive_slices": multi_component_slices,
            "adjacent_both_positive_pairs": len(case_adjacent_dice),
            "adjacent_dice_median": float(np.median(case_adjacent_dice)) if case_adjacent_dice else None,
            "area_change_gt_2x_pairs": large_area_change_pairs,
            "positive_run_lengths": runs,
            "geometry_match": same_geometry,
            "infection_encoding_match": infection_match,
            "lung_or_infection_union_match": union_match,
        }
        rows.append(row)
        per_case.append({"id": identifier, "areas_by_slice": case_areas})
        del inf, lung_data, infection_binary, lung_binary

    with (OUT / "case_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    intensity = {}
    for source, arrays in intensity_by_source.items():
        values = np.concatenate(arrays)
        intensity[source] = {
            "sample_count": int(values.size),
            "sample_stride": [8, 8, 4],
            "percentiles": percentile(values, (0, 1, 5, 50, 95, 99, 100)),
        }
    summary = {
        "cases": len(rows), "slices": sum(x["slices"] for x in rows),
        "source_cases": dict(Counter(x["source"] for x in rows)),
        "source_slices": dict(Counter({source: sum(x["slices"] for x in rows if x["source"] == source)
                                      for source in intensity})),
        "shape_values": dict(Counter(x["shape"] for x in rows)),
        "orientation_values": dict(Counter(x["orientation"] for x in rows)),
        "spacing_z_percentiles": percentile([x["spacing_z_mm"] for x in rows]),
        "intensity": intensity,
        "positive_slices": sum(x["positive_slices"] for x in rows),
        "empty_slices": sum(x["empty_slices"] for x in rows),
        "total_voxels": sum(x["all_voxels"] for x in rows),
        "infection_voxels": sum(x["infection_voxels"] for x in rows),
        "lung_voxels": sum(x["lung_voxels"] for x in rows),
        "infection_outside_lung": sum(x["infection_outside_lung"] for x in rows),
        "combined_label_counts": dict(label_counts),
        "infection_encoding_match_all": infection_encoding_equal,
        "lung_or_infection_union_match_all": combined_union_equal,
        "geometry_match_all": geometry_equal,
        "positive_slice_area_px_percentiles": percentile(all_areas),
        "component_2d_count": total_components,
        "component_2d_area_px_percentiles": percentile(all_components),
        "component_2d_under_10px": sum(x < 10 for x in all_components),
        "component_2d_under_50px": sum(x < 50 for x in all_components),
        "component_2d_under_100px": sum(x < 100 for x in all_components),
        "multicomponent_positive_slices": sum(x["multicomponent_positive_slices"] for x in rows),
        "adjacent_both_positive_pairs": len(adjacent_dice),
        "adjacent_dice_percentiles": percentile(adjacent_dice),
        "positive_run_length_percentiles": percentile(all_runs),
        "positive_runs": len(all_runs),
        "isolated_positive_runs": sum(x == 1 for x in all_runs),
        "large_area_change_pairs": sum(x["area_change_gt_2x_pairs"] for x in rows),
        "nearest_neighbor_resize": {
            str(k): {"lost_components": total_lost_components[k],
                     "lost_positive_slices": total_lost_positive_slices[k],
                     "total_components": total_components,
                     "total_positive_slices": total_gt_positive_slices}
            for k in (128, 256, 352)
        },
        "per_case_slice_areas": per_case,
    }
    summary["infection_fraction_all"] = summary["infection_voxels"] / summary["total_voxels"]
    summary["infection_fraction_lung"] = summary["infection_voxels"] / summary["lung_voxels"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_case_slice_areas"}, indent=2))


if __name__ == "__main__":
    main()
