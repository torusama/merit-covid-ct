"""Read-only, bounded-memory training-cohort intensity and XY-retention audit.

This is diagnostic streaming over NiBabel proxies, not a replacement training
reader. It reuses the shared header, orientation, voxel and split validation.
No normalization, HU window, split generation or label-based input selection.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import label
import yaml

from .cases import load_cases
from .config import PreprocessConfig
from .geometry import inspect_pair, _lps_transform
from .preprocessing import validate_voxels
from .splits import validate_splits


PERCENTILES = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
HIST_RANGE = (-32768, 65536)
HIST_BINS = HIST_RANGE[1] - HIST_RANGE[0]


def dump_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def dump_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_scope(config_path, fold=0):
    """Resolve only metadata; never construct validation/test voxel Datasets."""
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be 0..4")
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = config["data"]
    manifest = (path.parent / data["manifest"]).resolve()
    split_path = (path.parent / data["split_path"]).resolve()
    patient_map = (path.parent / data["patient_map"]).resolve() if data.get("patient_map") else None
    cases, identity = load_cases(manifest, patient_map=patient_map)
    raw = split_path.read_bytes()
    split = json.loads(raw.decode("utf-8-sig"))
    validate_splits(split, cases)
    by_id = {c.case_id: c for c in cases}
    overlaps = []
    for item in split["folds"]:
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            a, b = item[left], item[right]
            overlaps.append({"fold": item["fold"], "left": left, "right": right,
                "case_overlap": len(set(a["case_ids"]) & set(b["case_ids"])),
                "group_overlap": len(set(a["group_ids"]) & set(b["group_ids"])),
                "image_path_overlap": len({str(by_id[x].image.resolve()) for x in a["case_ids"]}
                    & {str(by_id[x].image.resolve()) for x in b["case_ids"]})})
    part = next(f for f in split["folds"] if f["fold"] == fold)
    selected = [by_id[x] for x in part["train"]["case_ids"]]
    provenance = {"fold": fold, "partition": "train", "selected_case_ids": [c.case_id for c in selected],
        "manifest": str(manifest), "split_path": str(split_path), "identity": identity,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "split_fingerprint": split["manifest_fingerprint"], "overlaps": overlaps,
        "fold_counts": [{"fold": f["fold"], **{r: len(f[r]["case_ids"]) for r in ("train", "validation", "test")}}
                        for f in split["folds"]],
        "group_disjoint_status": "PASS", "patient_independence_status": "NOT_INDEPENDENTLY_VERIFIED",
        "patient_note": identity["identity_note"],
        "voxel_access_policy": "Only the selected training cases; validation/test metadata only",
        "content_duplicate_check": "NOT_RUN: path/group checks do not prove absence of duplicate voxels or repeated clinical patients"}
    return config, selected, cases, split, provenance


class IntensityAccumulator:
    """Exact population moments plus all-voxel histogram, one native unit/bin."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = 0.0
        self.minimum, self.maximum = float("inf"), -float("inf")
        self.histogram = np.zeros(HIST_BINS, dtype=np.int64)
        self.underflow = self.overflow = 0
        self.noninteger = 0

    def update(self, values):
        values = np.asarray(values)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Statistics require nonempty finite scaled values")
        n = values.size
        mean = float(np.mean(values, dtype=np.float64))
        m2 = float(np.var(values, dtype=np.float64)) * n
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total
        self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.underflow += int(np.count_nonzero(values < HIST_RANGE[0]))
        self.overflow += int(np.count_nonzero(values >= HIST_RANGE[1]))
        self.noninteger += int(np.count_nonzero(values != np.floor(values)))
        bounded = values
        if values.min() < HIST_RANGE[0] or values.max() >= HIST_RANGE[1]:
            bounded = values[(values >= HIST_RANGE[0]) & (values < HIST_RANGE[1])]
        self.histogram += np.histogram(bounded, bins=HIST_BINS, range=HIST_RANGE)[0]

    def result(self):
        if not self.count:
            raise ValueError("No voxels accumulated")
        cdf = np.cumsum(self.histogram)
        quantiles = {}
        for q in PERCENTILES:
            rank = (self.count - 1) * q / 100
            if np.floor(rank) < self.underflow or np.ceil(rank) >= self.count - self.overflow:
                value = None
            else:
                a = np.searchsorted(cdf, np.floor(rank)-self.underflow, side="right") + HIST_RANGE[0]
                b = np.searchsorted(cdf, np.ceil(rank)-self.underflow, side="right") + HIST_RANGE[0]
                value = float(a + (b-a)*(rank-np.floor(rank)))
            quantiles[f"p{str(q).replace('.', '_')}"] = value
        return {"voxel_count": self.count, "min": self.minimum, "max": self.maximum,
            "mean": self.mean, "std_population": float(np.sqrt(self.m2 / self.count)),
            **quantiles, "percentile_method": "all-voxel linear histogram quantiles; exact for integer-valued data, <1 unit bin error for fractional data",
            "histogram_underflow": self.underflow, "histogram_overflow": self.overflow,
            "noninteger_voxels": self.noninteger}


def nearest_indices(source, target):
    if type(target) is not int or target < 1:
        raise ValueError("Target size must be a positive integer")
    return np.minimum(((np.arange(target) + 0.5) * source / target).astype(int), source - 1)


def resize_plane_audit(plane, size):
    """Measure native support, never subtract raw counts on unequal XY grids."""
    plane = np.asarray(plane, dtype=bool)
    h, w = plane.shape
    labels, count = label(plane)  # Same 2D 4-connectivity as shared retention QC.
    down = plane[np.ix_(nearest_indices(h, size), nearest_indices(w, size))]
    selected_labels = labels[np.ix_(nearest_indices(h, size), nearest_indices(w, size))]
    surviving = np.unique(selected_labels)
    surviving = surviving[surviving > 0]
    lost_ids = np.setdiff1d(np.arange(1, count + 1), surviving, assume_unique=True)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    up = down[np.ix_(nearest_indices(size, h), nearest_indices(size, w))]
    native_count = int(plane.sum())
    stats = {"size": size, "native_lesion_voxels": native_count,
        "native_components_2d": int(count), "lost_components_2d": int(len(lost_ids)),
        "lesion_voxels_in_lost_components": int(areas[lost_ids].sum()),
        "native_lesion_voxels_lost_roundtrip": int(np.count_nonzero(plane & ~up)),
        "native_background_voxels_gained_roundtrip": int(np.count_nonzero(~plane & up)),
        "positive_slices": int(native_count > 0), "positive_to_empty_slices": int(native_count > 0 and not down.any()),
        "resized_foreground_pixels": int(down.sum())}
    return stats, down, up


def native_blocks(case, assume_mm=False, block_depth=8):
    """Sequential scaled proxy reads; bounded even for large compressed CTs.

    Current cohort stores axial stacks as source axis 2. Reject another layout
    rather than silently slicing the wrong axis. XY permutations/flips and LPI
    Z reversal are handled by the same NiBabel transform as the shared reader.
    """
    config = PreprocessConfig(assume_mm=assume_mm)
    image, mask = inspect_pair(case, config)
    transform = _lps_transform(image)
    if int(transform[2, 0]) != 2:
        raise ValueError("Streaming audit requires source axis 2 to remain canonical Z")
    # Keep gzip handles open so increasing source-Z reads do not restart decoding.
    image = nib.load(str(case.image), keep_file_open=True)
    mask = nib.load(str(case.mask), keep_file_open=True)
    depth = image.shape[2]
    for start in range(0, depth, block_depth):
        stop = min(start + block_depth, depth)
        ct = np.asarray(image.dataobj[:, :, start:stop], dtype=np.float32)
        gt = np.asarray(mask.dataobj[:, :, start:stop], dtype=np.float32)
        validate_voxels(ct, gt, case.case_id)
        ct = nib.orientations.apply_orientation(ct, transform).transpose(2, 1, 0)
        gt = nib.orientations.apply_orientation(gt, transform).transpose(2, 1, 0)
        first = start if transform[2, 1] > 0 else depth - stop
        yield range(first, first + stop - start), ct, gt


def _snapshot(paths):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def _plots(output, histogram, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(HIST_BINS) + HIST_RANGE[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax in axes:
        ax.plot(x, histogram, linewidth=0.7)
        ax.set_yscale("log")
        ax.set(xlabel="Native NIfTI-scaled intensity (HU not independently verified)", ylabel="Voxel count (log)")
    axes[0].set_title("Full intensity range")
    axes[1].set(xlim=(-2200, 3200), title="Display-only zoom; no input clipping")
    fig.tight_layout()
    fig.savefig(output / "intensity_histogram.png", dpi=140)
    plt.close(fig)
    for i, sample in enumerate(examples):
        fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
        ct, gt = sample["image"], sample["mask"]
        low, high = np.percentile(ct, (2, 98))
        axes[0].imshow(ct, cmap="gray", vmin=low, vmax=high)
        axes[0].set_title("Native CT (display only)")
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Native GT")
        for j, size in enumerate((128, 256)):
            stats, down, up = resize_plane_audit(gt, size)
            axes[2+j*2].imshow(down, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[2+j*2].set_title(f"GT at {size}: {stats['lost_components_2d']} regions lost")
            axes[3+j*2].imshow(ct, cmap="gray", vmin=low, vmax=high)
            axes[3+j*2].imshow(np.ma.masked_equal(gt & ~up, 0), cmap="autumn_r", vmin=0, vmax=1, alpha=.9)
            axes[3+j*2].set_title(f"{size}: native FN support in red")
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"TRAIN ONLY {sample['case_id']} canonical z={sample['slice']} | diagnostic GT-selected illustration, not input selection")
        fig.tight_layout()
        fig.savefig(output / f"resize_example_{i+1}.png", dpi=160)
        plt.close(fig)


def inspect_training(config_path, output, *, fold=0, sizes=(128, 256), assume_mm=False, block_depth=8, limit=None):
    """Audit only selected training voxels; all-fold leakage is metadata-only."""
    sizes = tuple(sizes)
    if not sizes or len(set(sizes)) != len(sizes) or any(type(s) is not int or s < 1 for s in sizes):
        raise ValueError("Unique positive sizes required")
    if type(block_depth) is not int or block_depth < 1 or (limit is not None and limit < 1):
        raise ValueError("Positive block_depth/limit required")
    _, selected, _, _, provenance = audit_scope(config_path, fold)
    if limit is not None:
        selected = selected[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [c.image for c in selected] + [c.mask for c in selected] + [Path(provenance["manifest"]), Path(provenance["split_path"])]
    before = _snapshot(paths)
    global_stats = IntensityAccumulator()
    intensity_rows, resize_rows, slice_rows, failures, examples = [], [], [], [], []
    start_time = time.perf_counter()
    for i, case in enumerate(selected):
        stats = IntensityAccumulator()
        totals = {size: Counter() for size in sizes}
        best = None
        case_slices = []
        try:
            header = nib.load(str(case.image))
            for indices, image, mask in native_blocks(case, assume_mm, block_depth):
                stats.update(image)
                for z, ct, gt in zip(indices, image, mask):
                    for size in sizes:
                        row, _, _ = resize_plane_audit(gt, size)
                        case_slices.append({"case_id": case.case_id, "canonical_slice": z, **row})
                        totals[size].update({k: v for k, v in row.items() if k != "size"})
                        score = row["native_lesion_voxels_lost_roundtrip"] if size == min(sizes) else -1
                        if best is None or score > best["score"]:
                            best = {"case_id": case.case_id, "slice": z, "score": score,
                                    "image": ct.copy(), "mask": (gt > 0).copy()}
            intensity_rows.append({"case_id": case.case_id, "group_id": case.group_id,
                "stored_dtype": str(header.get_data_dtype()), "slope": float(header.dataobj.slope),
                "intercept": float(header.dataobj.inter), "spatial_unit": header.header.get_xyzt_units()[0],
                "units_assumed_mm": header.header.get_xyzt_units()[0] == "unknown" and assume_mm,
                **stats.result()})
            global_stats.histogram += stats.histogram
            # Merge exact moments without rereading this CT.
            n = global_stats.count + stats.count
            delta = stats.mean - global_stats.mean
            global_stats.m2 += stats.m2 + delta**2 * global_stats.count * stats.count / n
            global_stats.mean += delta * stats.count / n
            global_stats.count = n
            global_stats.minimum = min(global_stats.minimum, stats.minimum)
            global_stats.maximum = max(global_stats.maximum, stats.maximum)
            global_stats.underflow += stats.underflow
            global_stats.overflow += stats.overflow
            global_stats.noninteger += stats.noninteger
            for size, values in totals.items():
                lost = values["native_lesion_voxels_lost_roundtrip"]
                native = values["native_lesion_voxels"]
                resize_rows.append({"case_id": case.case_id, "size": size, **dict(values),
                    "native_voxel_loss_fraction_roundtrip": lost/native if native else None,
                    "lost_component_voxel_fraction": values["lesion_voxels_in_lost_components"]/native if native else None})
            slice_rows.extend(case_slices)
            examples.append(best)
            examples = sorted(examples, key=lambda x: (-x["score"], x["case_id"]))[:3]
            print(f"audit {i+1}/{len(selected)} {case.case_id}: range={stats.minimum:g}..{stats.maximum:g} "
                  f"lost_components={{{', '.join(str(s)+':'+str(totals[s]['lost_components_2d']) for s in sizes)}}}", flush=True)
        except (ValueError, OSError, MemoryError) as error:
            failures.append({"case_id": case.case_id, "error": str(error)})
            print(f"audit {i+1}/{len(selected)} FAIL {case.case_id}: {error}", flush=True)
    aggregate = []
    for size in sizes:
        rows = [r for r in resize_rows if r["size"] == size]
        total = {k: sum(r[k] for r in rows) for k in (
            "native_lesion_voxels", "native_components_2d", "lost_components_2d", "lesion_voxels_in_lost_components",
            "native_lesion_voxels_lost_roundtrip", "native_background_voxels_gained_roundtrip",
            "positive_slices", "positive_to_empty_slices", "resized_foreground_pixels")}
        native = total["native_lesion_voxels"]
        aggregate.append({"size": size, **total,
            "native_voxel_loss_fraction_roundtrip": total["native_lesion_voxels_lost_roundtrip"]/native if native else None,
            "lost_component_voxel_fraction": total["lesion_voxels_in_lost_components"]/native if native else None,
            "cases_losing_components": sum(r["lost_components_2d"] > 0 for r in rows)})
    modified = [p for p, stamp in before.items() if _snapshot([Path(p)])[p] != stamp]
    report = {"status": "PASS" if not failures and not modified else "FAIL", "provenance": provenance,
        "requested_cases": len(selected), "audited_cases": len(intensity_rows), "failures": failures,
        "scope": "train-only native NIfTI-scaled float32 voxels; no intensity window/normalization applied",
        "assume_mm": assume_mm, "unit_note": "Only spatial metadata; intensity/label pixel counts do not establish mm or HU",
        "hu_verification": {"status": "SOURCE_DOCUMENTED_NOT_INDEPENDENTLY_DICOM_VERIFIED",
            "evidence": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC9444848/",
                         "https://huggingface.co/datasets/MedOtter/COVID-19-20"],
            "limitations": "Source describes HU. Histogram/range/slope alone cannot prove per-case calibration; no per-case DICOM identity mapping available."},
        "intensity": global_stats.result() if global_stats.count else None, "resize": aggregate,
        "resize_definitions": {"components": "2D 4-connected; completely absent under nearest-exact downsampling",
            "voxel_loss": "Native GT foreground missing after nearest-exact downsample then nearest-exact restore to original XY",
            "lost_component_voxels": "Native voxels belonging to entirely lost 2D components",
            "warning": "Do not subtract pixel counts on differently sized grids; roundtrip loss is not a model performance bound"},
        "source_files_modified": modified, "seconds": time.perf_counter()-start_time}
    dump_csv(output / "intensity_cases.csv", intensity_rows)
    dump_csv(output / "resize_cases.csv", resize_rows)
    dump_csv(output / "resize_slices.csv", slice_rows)
    dump_csv(output / "resize_summary.csv", aggregate)
    dump_csv(output / "split_overlap.csv", provenance["overlaps"])
    dump_csv(output / "intensity_histogram.csv", ({"left_edge": i+HIST_RANGE[0], "voxel_count": int(n)}
             for i, n in enumerate(global_stats.histogram) if n))
    _plots(output, global_stats.histogram, examples)
    dump_json(output / "audit.json", report)
    return report
