"""Resolve original and Kaggle-extracted NIfTI layouts without changing cases."""
from pathlib import Path


def case_paths(case_id, kind):
    stem = f"{case_id}_{kind}"
    return (f"{stem}.nii.gz", f"{stem}.nii", f"{stem}.nii/{case_id}.nii",
            f"{stem}.nii/{stem}.nii")


def resolve_case_files(train, case_id, available):
    result = {}
    for key, kind in (("image", "ct"), ("mask", "seg")):
        matches = [name for name in case_paths(case_id, kind) if name in available]
        if len(matches) != 1:
            raise ValueError(f"{case_id} {key}: expected one file, found {matches}")
        result[key] = Path(train) / matches[0]
    return result
