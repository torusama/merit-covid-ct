"""Kiểm tra điểm lấy input mới: ba partition của cùng một fold, không training."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ct_pipeline import PreprocessConfig, build_fold_loaders


def test_fold_loaders_keep_patient_groups_disjoint_and_seeded(grouped_split):
    config = PreprocessConfig(slab_size=3, size_hw=(12, 16))
    loaders = build_fold_loaders(
        *grouped_split, fold=0, preprocess=config, batch_size=2, seed=7
    )
    repeated = build_fold_loaders(
        *grouped_split, fold=0, preprocess=config, batch_size=2, seed=7
    )
    partition_groups = []
    all_case_ids = []
    for name in ("train", "validation", "test"):
        dataset = getattr(loaders, name).dataset
        assert dataset.config == config
        assert dataset.partition == name
        partition_groups.append({case.group_id for case in dataset.cases})
        all_case_ids.extend(case.case_id for case in dataset.cases)
    assert len(all_case_ids) == len(set(all_case_ids)) == 10
    for first in range(3):
        for second in range(first + 1, 3):
            assert not partition_groups[first] & partition_groups[second]

    batch = next(iter(loaders.train))
    replay = next(iter(repeated.train))
    assert batch["image_slab"].shape == (2, 1, 3, 12, 16)
    assert batch["master_mask"].shape == (2, 1, 12, 16)
    assert batch["z_offsets_mm"].shape == (2, 3)
    assert batch["case_id"] == replay["case_id"]
    assert torch.equal(batch["image_slab"], replay["image_slab"])


def test_fold_loaders_reject_tampered_partition(grouped_split):
    manifest, path = grouped_split
    split = json.loads(path.read_text(encoding="utf-8"))
    fold = split["folds"][0]
    fold["train"]["case_ids"].append(fold["test"]["case_ids"][0])
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError):
        build_fold_loaders(manifest, path, fold=0)


def test_inspect_example_works_outside_repo_and_reads_test_partition(grouped_split, tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples/read_training_batch.py"
    manifest, split = grouped_split
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest", str(manifest),
            "--split", str(split),
            "--partition", "test",
            "--size", "12", "16",
            "--slab-size", "3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["mode"] == "inspect_batch_only"
    assert report["device"] == "cpu"
    assert report["selected_partition"] == "test"
    assert report["volumes_in_partition"] == {"train": 6, "validation": 2, "test": 2}
    assert report["inputs"]["shape"] == [2, 1, 3, 12, 16]
    assert report["targets"]["shape"] == [2, 1, 12, 16]
    assert report["master_indices"] == [0, 1]
