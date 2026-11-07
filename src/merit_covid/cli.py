"""Small command implementations; scripts only dispatch into this package."""

import argparse
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

from .data.manifest import load_manifest, write_manifest
from .data.nifti_io import load_case, validate_image_header, verify_alignment
from .data.splitting import make_splits, save_splits


def check_dataset_main():
    parser = argparse.ArgumentParser(description="Check manifest paths and paired NIfTI geometry; never modify source data")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset", choices=("covid19_20", "zenodo20"))
    parser.add_argument("--headers-only", action="store_true", help="Skip voxel/label validation; report the limited scope")
    parser.add_argument("--limit", type=int, help="Explicit small-case smoke check")
    parser.add_argument("--assumed-unit", choices=("mm",), help="Explicit agreed mm assumption for unknown NIfTI units")
    parser.add_argument("--write-manifest", type=Path, help="Create a separate versioned manifest; never overwrite")
    args = parser.parse_args()
    cases = load_manifest(args.manifest, dataset=args.dataset)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    selected = cases if args.limit is None else cases[:args.limit]
    enriched = []
    for case in selected:
        image = nib.load(str(case.image))
        _, scale = validate_image_header(case, image, assumed_unit=args.assumed_unit)
        if case.infection_mask is not None:
            verify_alignment(image, nib.load(str(case.infection_mask)))
        if case.lung_mask is not None:
            verify_alignment(image, nib.load(str(case.lung_mask)))
        spacing = tuple(float(v) for v in nib.affines.voxel_sizes(image.affine) * scale)
        if not args.headers_only:
            load_case(case, require_mask=case.infection_mask is not None, assumed_unit=args.assumed_unit)
        enriched.append(replace(case, shape=tuple(image.shape), spacing_mm=spacing,
                                affine=tuple(tuple(float(v) for v in row) for row in image.affine),
                                orientation=tuple(nib.aff2axcodes(image.affine))))
        print(f"OK {case.dataset}/{case.case_id}: shape={image.shape}, spacing_mm={spacing}, intensity={case.intensity}")
    if args.write_manifest:
        write_manifest(enriched, args.write_manifest)
    scope = "headers/paths only; voxel labels NOT checked" if args.headers_only else "geometry, finite CT and infection labels"
    print(f"Checked {len(selected)}/{len(cases)} cases ({scope}). No intensity provenance inferred.")


def make_splits_main():
    parser = argparse.ArgumentParser(description="Create immutable primary-only grouped train/validation/internal-test folds")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--expected-count", type=int, default=199)
    args = parser.parse_args()
    cases = load_manifest(args.manifest)
    payload = make_splits(cases, seed=args.seed, num_folds=args.num_folds, expected_count=args.expected_count)
    save_splits(payload, args.output)
    print(f"Created {args.num_folds} folds for {len(cases)} primary cases: {args.output}")
    if any(c.patient_id is None for c in cases):
        print("Patient identity is unverified for some cases: volume-level fallback is recorded; verify linkage before publication.")


def smoke_test_main():
    """Create temporary synthetic NIfTI and verify the complete data/evaluation path."""
    from tempfile import TemporaryDirectory
    from .config import Config
    from .data.dataset import SlabDataset
    from .data.manifest import Case
    from .data.nifti_io import save_prediction
    from .data.preprocessing import restore_prediction
    from .engine.metrics import binary_metrics
    from .engine.reconstruction import reconstruct_volume

    with TemporaryDirectory(prefix="merit-smoke-") as directory:
        root = Path(directory)
        values = np.arange(8 * 9 * 4, dtype=np.float32).reshape(8, 9, 4)
        mask = np.zeros(values.shape, dtype=np.uint8)
        mask[2:4, 3:5, 1:3] = 1
        for name, data in (("ct", values), ("mask", mask)):
            image = nib.Nifti1Image(data, np.diag([-0.7, 0.8, -5.0, 1.0]))
            image.header.set_xyzt_units("mm")
            nib.save(image, root / f"{name}.nii.gz")
        case = Case("covid19_20", "synthetic", root / "ct.nii.gz", root / "mask.nii.gz")
        dataset = SlabDataset([case], Config(), mode="validation")
        samples = [dataset[i] for i in range(len(dataset))]
        # Ground truth identity round-trip is a geometry check, not model inference.
        prepared = dataset.prepared_volume(0)
        reconstructed = reconstruct_volume(((s["slice_index"], s["mask"][0]) for s in reversed(samples)), prepared.image.shape)
        restored = restore_prediction(reconstructed, prepared)
        assert np.array_equal(restored, mask)
        assert binary_metrics(restored, mask)["dice"] == 1
        save_prediction(restored, prepared.volume, root / "roundtrip.nii.gz")
        assert np.array_equal(np.asanyarray(nib.load(root / "roundtrip.nii.gz").dataobj), mask)
        print("PASS: synthetic NIfTI -> Dataset -> slabs -> out-of-order reconstruction -> original geometry -> Dice=1")
        print("No model, real dataset, training, download or GPU was used.")
