"""Train-only fitted normalization and native-grid research metric checks."""
from copy import deepcopy

import nibabel as nib
import numpy as np
import pytest
import torch

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.metrics import lesion_slice_metrics, presence_calibration
from merit_covid.data.pipeline.preprocessing import normalize_ct
from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import load_development_split, split_datasets, validate_experiment
from scripts.fit_train_zscore import fit_train_zscore


def test_native_lesion_slices_and_presence_calibration():
    gt = np.zeros((3, 4, 4), dtype=np.uint8)
    gt[1, 0, 0] = gt[1, 3, 3] = 1
    pred = np.zeros_like(gt, dtype=float)
    pred[0, 0, 0] = pred[1, 0, 0] = 0.9
    result = lesion_slice_metrics(pred, gt, presence=np.array([0.2, 0.8, 0.1]))
    assert result["empty_gt_slices"] == 2 and result["empty_slices_with_fp"] == 1
    assert result["gt_components_3d"] == 2 and result["detected_components_3d"] == 1
    assert result["component_size_bins"]["lt_100_voxels"]["recall"] == 0.5
    assert result["presence"]["auroc"] == 1
    assert result["presence"]["brier"] == pytest.approx(0.03)
    empty = lesion_slice_metrics(np.zeros_like(pred), np.zeros_like(gt), presence=np.zeros(3))
    assert empty["presence"]["auroc"] is None
    assert empty["component_size_bins"]["lt_100_voxels"]["recall"] is None
    with pytest.raises(ValueError):
        presence_calibration([1.2], [1])


def test_train_fit_is_frozen_and_rejects_fold_or_case_leakage(tmp_path, monkeypatch):
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    cases, split, _, _ = load_development_split(config)
    train_ids = set(split["folds"][0]["train"]["case_ids"])
    allowed = {str(c.image) for c in cases if c.case_id in train_ids}
    real_load = nib.load
    def guarded(path, *args, **kwargs):
        assert str(path) in allowed, f"unexpected validation/test/GT voxel read: {path}"
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(nib, "load", guarded)
    params = fit_train_zscore(config, 0, block_depth=2)
    monkeypatch.setattr(nib, "load", real_load)
    train_arrays = [real_load(c.image).get_fdata(dtype=np.float32).ravel() for c in cases if c.case_id in train_ids]
    values = np.concatenate(train_arrays)
    assert params["zscore_mean"] == pytest.approx(float(values.mean()), abs=1e-6)
    assert params["zscore_std"] == pytest.approx(float(values.std()), abs=1e-6)
    config["preprocess"].update(intensity_encoding="train_zscore", **params)
    config = validate_experiment(config)
    datasets, _ = split_datasets(config, 0)
    assert all(d.config.zscore_mean == params["zscore_mean"] for d in datasets.values())
    x = np.array([-1e4, 0, 1e4], dtype=np.float32)
    y = normalize_ct(x, PreprocessConfig(**config["preprocess"]), "test")
    np.testing.assert_allclose(y, (x - params["zscore_mean"]) / params["zscore_std"], rtol=1e-6)
    with pytest.raises(ValueError, match="provenance"):
        split_datasets(config, 1)
    bad = deepcopy(config)
    bad["preprocess"]["zscore_fit"]["case_ids"].append("heldout")
    with pytest.raises(ValueError, match="provenance"):
        split_datasets(bad, 0)
    with pytest.raises(ValueError, match="std"):
        PreprocessConfig(intensity_encoding="train_zscore", zscore_mean=0, zscore_std=0, zscore_fit=params["zscore_fit"])


def test_presence_adapter_keeps_master_values():
    from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITUNETRPPAdapter
    torch.set_num_threads(2)
    model = MERITUNETRPP(img_size=(3, 32, 32), feature_size=2, depths=(1, 1, 1, 1)).eval()
    batch = {"image_slab": torch.zeros(1, 1, 3, 32, 32), "master_index": [1]}
    with torch.no_grad():
        logits, presence = MERITUNETRPPAdapter(model).predict_with_presence(batch)
        output = model(batch["image_slab"], [1])
    torch.testing.assert_close(presence, output.presence[:, 1])
    torch.testing.assert_close(logits, output.logits[:, 1:2, 1] - output.logits[:, :1, 1])


def test_evaluation_reports_native_slice_and_presence_metrics(tmp_path):
    from merit_covid.engine.training import evaluate_model
    from merit_covid.models.merit_unetrpp import MERITUNETRPP
    torch.set_num_threads(2)
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    datasets, _ = split_datasets(config, 0)
    model = MERITUNETRPP(**config["model"]).eval()
    report = evaluate_model(model, datasets["validation"].cases[:1], PreprocessConfig(**config["preprocess"]),
                            include_hd95=False)
    detail = report["per_case"][0]["lesion_slice_detail"]
    assert detail["empty_gt_slices"] >= 0 and detail["presence"] is not None
    assert 0 <= detail["presence"]["brier"] <= 1
