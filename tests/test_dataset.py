from dataclasses import replace
import importlib.util

import numpy as np
import pytest

from merit_covid.config import Config
from merit_covid.data.dataset import SlabDataset, make_dataloader


def test_numpy_dataset_modes_and_augmentation(nifti_case):
    case, _, _ = nifti_case()
    dataset = SlabDataset([case], Config(), mode="validation")
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["image"].shape == (1, 11, 8, 6)
    assert sample["mask"].shape == (1, 8, 6)
    assert sample["case_id"] == "case1"
    np.testing.assert_array_equal(dataset[0]["image"], sample["image"])
    with pytest.raises(ValueError, match="augmentation"):
        SlabDataset([case], Config(), mode="validation", augmentation=lambda s: s)
    train = SlabDataset([case], Config(), mode="train", augmentation=lambda s: dict(s, augmented=True))
    assert train[0]["augmented"]
    inference = SlabDataset([case], Config(), mode="inference")
    assert "mask" not in inference[0]
    # Inference ignores even an inaccessible recorded ground-truth path.
    absent = replace(case, infection_mask=case.infection_mask.with_name("absent.nii"))
    assert "mask" not in SlabDataset([absent], Config(), mode="inference")[0]


def test_external_isolation(nifti_case):
    case, _, _ = nifti_case(dataset="zenodo20")
    for mode in ("train", "validation", "internal_test"):
        with pytest.raises(ValueError, match="external only"):
            SlabDataset([case], Config(), mode=mode)
    assert SlabDataset([case], Config(), mode="external_test")[0]["mask"].shape == (1, 8, 6)


def test_missing_training_dependency(nifti_case, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    case, _, _ = nifti_case()
    with pytest.raises(ImportError, match="optional training"):
        make_dataloader(SlabDataset([case], Config(), mode="train"))


def test_torch_dataloader_when_available(nifti_case):
    torch = pytest.importorskip("torch")
    case, _, _ = nifti_case()
    loader = make_dataloader(SlabDataset([case], Config(batch_size=2), mode="validation"))
    batch = next(iter(loader))
    assert tuple(batch["image"].shape) == (2, 1, 11, 8, 6)
    assert tuple(batch["mask"].shape) == (2, 1, 8, 6)
    assert batch["image"].dtype == torch.float32
