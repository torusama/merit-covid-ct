"""End-to-end software regressions; all NIfTI/labels here are synthetic."""

from copy import deepcopy
from dataclasses import replace
import json
import math
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from merit_covid.data.pipeline.cases import Case, load_cases
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.data.pipeline.slab import sample_arrays, select_slices
from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.pipeline.splits import create_splits, validate_splits
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITUNETRPPAdapter, MERITLoss, MERITOutput
from merit_covid.engine.slab_inference import predict_slab_case
from merit_covid.engine.patch_inference import predict_patch_batch
from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import Trainer, split_datasets, validate_experiment, load_checkpoint_model, evaluate_model


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def metadata(spacing=1.0, depth=25):
    return {"canonical_shape_xyz": [4, 6, depth],
            "canonical_affine_mm": np.diag([-1., -1., spacing, 1]).tolist()}


@pytest.mark.parametrize("spacing", [1., 1.5, 3., 5.])
@pytest.mark.parametrize("master", [0, 12, 24])
def test_physical_sampling_matches_world_nearest_and_index_baseline(spacing, master):
    meta = metadata(spacing)
    native = np.arange(25) * spacing
    indices, requested, valid = select_slices(master, 25, 11, meta, sampling_mode="physical_mm", target_spacing_mm=1.5)
    # Independent brute-force world-distance reference with lower-index ties.
    expected = [min(range(25), key=lambda i: (round(abs(native[i]-native[master]-offset), 7), i)) for offset in requested]
    np.testing.assert_array_equal(indices, expected)
    assert indices[5] == master and valid[5] and np.all(np.diff(indices) >= 0)
    assert len(set(indices[valid])) == int(valid.sum())
    old, _, _ = select_slices(master, 25, 11, meta)
    np.testing.assert_array_equal(old, np.clip(np.arange(master-5, master+6), 0, 24))
    sample = sample_arrays(np.zeros((25, 6, 4), np.float32), None, master, 11, meta,
                           sampling_mode="physical_mm", target_spacing_mm=1.5)
    np.testing.assert_allclose(sample["z_offsets_mm"], (indices-master)*spacing)
    np.testing.assert_allclose(sample["requested_offsets_mm"], np.arange(-5, 6)*1.5)


def test_short_volume_master_priority_and_duplicate_policy():
    unique, _, valid = select_slices(0, 2, 11, metadata(5, 2))
    assert unique.tolist() == [0]*6 + [1]*5
    assert valid.tolist() == [False]*5 + [True, True] + [False]*4
    _, _, all_valid = select_slices(0, 2, 11, metadata(5, 2), duplicate_slice_policy="all")
    assert all_valid.all()
    with pytest.raises(ValueError, match="target_spacing"):
        PreprocessConfig(sampling_mode="physical_mm")
    with pytest.raises(ValueError, match="edge"):
        PreprocessConfig(sampling_mode="physical_mm", target_spacing_mm=2, boundary_policy="reflect")


@pytest.mark.parametrize("affine,unit", [
    (np.diag([.7, -.8, -3, 1]), "mm"),
    (np.array([[-.7, 0, .2, 4], [0, -.8, .3, 7], [0, 0, 3, 1], [0, 0, 0, 1]]), "mm"),
    (np.diag([-.0007, -.0008, .003, 1]), "meter"),
    (np.array([[0, 0, 3, 1], [-.7, 0, 0, 2], [0, -.8, 0, 3], [0, 0, 0, 1]]), "mm"),
])
def test_geometry_mm_sampling_roundtrip(nifti_case, affine, unit):
    source, _, mask = nifti_case(shape=(8, 10, 5), affine=affine, unit=unit)
    case = Case(source.case_id, source.image, source.infection_mask, source.case_id)
    config = PreprocessConfig(sampling_mode="physical_mm", target_spacing_mm=1.5)
    image, truth, meta = read_arrays(case, config)
    from merit_covid.data.pipeline.geometry import restore_volume
    np.testing.assert_array_equal(restore_volume(truth, meta), mask)
    sample = CTSlabDataset([case], config)[1]
    affine_mm = np.asarray(meta["canonical_affine_mm"])
    expected_step = np.linalg.norm(affine_mm[:3, 2])
    np.testing.assert_allclose(sample["z_offsets_mm"], (sample["slice_indices"]-1)*expected_step, atol=1e-5)


@pytest.mark.parametrize("batch_size,workers", [(1, 0), (2, 0), (2, 1)])
def test_full_gt_collation_and_windows_workers(nifti_case, batch_size, workers):
    source, _, _ = nifti_case(shape=(8, 10, 5), affine=np.diag([-1., -1., 3., 1.]))
    case = Case(source.case_id, source.image, source.infection_mask, source.case_id)
    dataset = CTSlabDataset([case], PreprocessConfig(size_hw=(32, 32), sampling_mode="physical_mm", target_spacing_mm=1.5))
    sample = dataset[1]
    _, labels, _ = dataset._load_case(0)
    torch.testing.assert_close(sample["mask_slab"][0], labels[sample["slice_indices"]])
    assert torch.equal(sample["master_mask"], sample["mask_slab"][:, 5])
    assert not torch.equal(sample["mask_slab"][:, 5], sample["mask_slab"][:, 9])
    batch = next(iter(make_dataloader(dataset, batch_size=batch_size, num_workers=workers)))
    assert batch["image_slab"].shape == batch["mask_slab"].shape == batch["valid_region"].shape == (batch_size, 1, 11, 32, 32)
    assert batch["master_index"].tolist() == [5]*batch_size
    assert batch["z_offsets_mm"].shape == batch["slice_valid"].shape == (batch_size, 11)
    assert len(batch["metadata"]) == batch_size and dataset.__getstate__()["_cached"] is None


def test_inference_never_opens_recorded_missing_gt_and_split_stays_valid(tmp_path):
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    path = config["data"]["split_path"]
    before = open(path, "rb").read()
    cases, _ = load_cases(config["data"]["manifest"])
    for case in cases:
        case.mask.unlink()
    dataset = CTSlabDataset.from_split(config["data"]["manifest"], path, 0, "test", require_mask=False)
    sample = dataset[0]
    assert "mask_slab" not in sample and "master_mask" not in sample
    assert open(path, "rb").read() == before
    with pytest.raises(FileNotFoundError):
        CTSlabDataset.from_split(config["data"]["manifest"], path, 0, "train")


def test_resize_reports_tiny_lost_component_and_preserves_binary(nifti_case):
    labels = np.zeros((64, 64, 4), dtype=np.uint8)
    labels[0, 0, 1] = 1  # Not sampled by nearest-exact 64 -> 32.
    source, _, _ = nifti_case(shape=labels.shape, labels=labels, affine=np.diag([-1., -1., 1., 1.]))
    case = Case(source.case_id, source.image, source.infection_mask, source.case_id)
    with pytest.warns(UserWarning, match="lost lesion"):
        _, mask, info = read_arrays(case, PreprocessConfig(size_hw=(32, 32)))
    assert set(np.unique(mask)) <= {0, 1}
    assert info["lesion_retention"]["lost_positive_slices"] == [1]
    assert info["lesion_retention"]["lost_components_by_slice"][0]["lost"] == 1


def test_physical_position_mode_explicit_and_index_unchanged():
    kwargs = dict(img_size=(11, 32, 32), feature_size=2, depths=(1, 1, 1, 1))
    index_model = MERITUNETRPP(**kwargs).eval()
    physical = MERITUNETRPP(**kwargs, position_mode="physical_mm", position_unit_mm=1.0).eval()
    physical.load_state_dict(index_model.state_dict())
    image = torch.randn(1, 1, 11, 32, 32)
    offsets = torch.arange(-5, 6)[None].float()
    with torch.no_grad():
        baseline = index_model(image, [5])
        same = index_model(image, [5], offsets*5)
        mm = physical(image, [5], offsets)
        farther = physical(image, [5], offsets*3)
    for a, b in zip(baseline, same):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(baseline.logits, mm.logits)
    assert not torch.equal(mm.logits, farther.logits)
    with pytest.raises(ValueError, match="z_offsets_mm"):
        physical(image, [5])
    with pytest.raises(ValueError, match="zero"):
        physical(image, [5], offsets+1)


@pytest.mark.parametrize("sigma", [1e-300, 2., 1e300])
def test_physical_loss_validity_extreme_sigma_and_alignment(sigma):
    logits = torch.randn(1, 2, 3, 4, 4, requires_grad=True)
    presence = torch.tensor([[.7, .6, .2]], requires_grad=True)
    salience = torch.tensor([[.3, .6, .2]], requires_grad=True)
    target = torch.zeros(1, 1, 3, 4, 4)
    target[:, :, 1, 0, 0] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :, 0] = False
    out = MERITOutput(logits, presence, salience, logits)
    criterion = MERITLoss(weighting_mode="physical_mm", sigma_mm=sigma, alignment_pool="top_k_mean")
    loss = criterion(out, target, [1], valid_region=valid, z_offsets_mm=[[-5, 0, 5]])
    assert all(torch.isfinite(v) for v in loss.values())
    loss["total"].backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="master"):
        criterion(out, target, [0], valid_region=valid, z_offsets_mm=[[0, 5, 10]])
    with pytest.raises(ValueError, match="z_offsets_mm"):
        criterion(out, target, [1], valid_region=valid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_amp_real_forward_loss_backward():
    model = MERITUNETRPP(img_size=(11,32,32), feature_size=2, depths=(1,1,1,1)).cuda()
    image = torch.randn(1,1,11,32,32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        result = model(image, [5])
        loss = MERITLoss()(result, torch.zeros_like(image), [5])
    loss["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_native_master_inference_two_class_conversion_and_patch_rejection(nifti_case):
    source, raw, _ = nifti_case(shape=(7,9,5), affine=np.diag([.7, .8, -3., 1.]))
    case = Case(source.case_id, source.image, source.infection_mask, source.case_id)
    model = MERITUNETRPP(img_size=(11,32,32), feature_size=2, depths=(1,1,1,1), pcg_ramp=0).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.out1.conv.conv.bias.copy_(torch.tensor([5.,6.]))
    source.infection_mask.unlink()
    result = predict_slab_case(model, case, PreprocessConfig(size_hw=(32,32)), batch_size=2)
    assert result.probability_zyx.shape == (5,9,7)
    np.testing.assert_allclose(result.to_original_xyz(), np.full(raw.shape, 1/(1+math.exp(-1))), rtol=1e-6)
    assert result.metadata["processed_size_hw"] == result.metadata["native_size_hw"]
    with pytest.raises(ValueError, match="predict_slab_case"):
        predict_patch_batch(model, {"image_patch": torch.zeros(1,1,64,32,32)})


@pytest.mark.parametrize("shuffle_mode", ["slice", "volume"])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_resume_matches_uninterrupted_training_and_rejects_protocol_drift(tmp_path, shuffle_mode, activation_checkpointing):
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    config["training"]["shuffle_mode"] = shuffle_mode
    config["training"]["activation_checkpointing"] = activation_checkpointing
    if activation_checkpointing:
        # Exercise the primary no-resize path with a native 32-grid fixture.
        # Pad source fixtures once here; the production reader must not resize.
        for path in (tmp_path / "data").glob("*.nii.gz"):
            image = nib.load(path)
            values = np.pad(image.get_fdata(dtype=np.float32), ((0, 16), (0, 14), (0, 0)))
            nib.save(nib.Nifti1Image(values, image.affine, image.header.copy()), path)
        config["preprocess"]["size_hw"] = None
    datasets, info = split_datasets(config, 0)
    def trainer(cfg, path):
        loader = make_dataloader(datasets["train"], batch_size=2, shuffle=True,
                                 shuffle_mode=shuffle_mode, seed=cfg["training"]["seed"])
        return Trainer(cfg, loader, datasets["validation"].cases, info, path)
    short = trainer(config, tmp_path / "resume")
    original = {name: p.detach().clone() for name, p in short.model.named_parameters()}
    history = short.fit()
    assert history[0]["pcg_ramp"] == 0 and history[0]["train"]["optimizer_steps"] > 0
    assert any(not torch.equal(original[n], p) for n, p in short.model.named_parameters())
    extended = deepcopy(config)
    extended["training"]["epochs"] = 2
    resumed = trainer(extended, tmp_path / "resume")
    continuation = resumed.fit(resume=tmp_path / "resume" / "last.pt")
    assert continuation[0]["epoch"] == 1
    whole = trainer(extended, tmp_path / "whole")
    whole.fit()
    for key, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    loaded, cp = load_checkpoint_model(tmp_path / "resume" / "last.pt")
    assert getattr(loaded, "_activation_checkpointing_enabled", False) == activation_checkpointing
    assert loaded.pcg_ramp.item() == resumed.model.pcg_ramp.item()
    report_path = tmp_path / "evaluated.json"
    subprocess.run([sys.executable, "scripts/merit_pipeline.py", "evaluate", "--checkpoint",
                    str(tmp_path / "resume" / "last.pt"), "--partition", "test", "--device", "cpu",
                    "--output", str(report_path)], check=True, capture_output=True, text=True)
    report = json.loads(report_path.read_text())
    assert len(report["per_case"]) == 2 and report["evaluation_grid"] == "full original NIfTI XYZ"
    assert report["protocol_fingerprint"] == cp["protocol_fingerprint"]
    drift = deepcopy(config)
    drift["training"]["activation_checkpointing"] = not activation_checkpointing
    with pytest.raises(ValueError, match="mismatch"):
        trainer(drift, tmp_path / "execution_drift").resume(tmp_path / "resume" / "last.pt")
    drift = deepcopy(config)
    drift["loss"]["alignment_pool"] = "top_k_mean"
    with pytest.raises(ValueError, match="mismatch"):
        trainer(drift, tmp_path / "drift").resume(tmp_path / "resume" / "last.pt")
    overlapping = datasets["train"].cases
    with pytest.raises(ValueError, match="leakage"):
        Trainer(config, short.train_loader, overlapping, info, tmp_path / "bad")


def test_split_validator_rejects_group_leakage(tmp_path):
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    cases, identity = load_cases(config["data"]["manifest"])
    split = create_splits(cases, identity)
    split["folds"][0]["validation"] = deepcopy(split["folds"][0]["train"])
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(split, cases)


def test_far_physical_targets_keep_one_real_endpoint_not_zero_supervision():
    indices, _, valid = select_slices(0, 2, 11, metadata(5, 2), sampling_mode="physical_mm", target_spacing_mm=20)
    assert set(indices[valid]) == {0, 1} and valid.sum() == 2 and valid[5]


def test_physical_gaussian_uses_mm_not_slice_units():
    logits = torch.tensor([0., 1., 2., 2., 0., -1.]).reshape(1, 2, 3, 1, 1)
    target = torch.zeros(1, 1, 3, 1, 1)
    out = MERITOutput(logits, None, None, logits)
    physical = MERITLoss(weighting_mode="physical_mm", sigma_mm=1.0)
    result = physical(out, target, [1], z_offsets_mm=[[-5., 0., 5.]])
    probability = logits.softmax(1)[:, 1].flatten()
    dice = 1 - physical.smooth / (probability + physical.smooth)
    ce = torch.nn.functional.cross_entropy(logits, target[:, 0].long(), reduction="none").flatten()
    weights = torch.exp(-.5 * torch.tensor([25., 0., 25.]))
    torch.testing.assert_close(result["segmentation"], ((dice+ce)*weights).sum()/weights.sum())
    index = MERITLoss()(out, target, [1])["segmentation"]
    assert not torch.isclose(result["segmentation"], index)


def test_primary_loader_rejects_external_legacy_manifest(tmp_path):
    directory = tmp_path / "zenodo20"
    directory.mkdir()
    path = directory / "cases.json"
    path.write_text('[{"id":"z","image":"x.nii.gz","mask":"y.nii.gz"}]')
    with pytest.raises(ValueError, match="external"):
        load_cases(path)


def test_physical_cli_requires_explicit_spacing_and_records_split(tmp_path):
    from merit_covid.engine.training import load_experiment
    config = synthetic_experiment(tmp_path / "data", sampling_mode="physical_mm")
    path = tmp_path / "physical.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="target_spacing_mm"):
        load_experiment(path)
    configured = load_experiment(path, target_spacing_mm=1.5, position_mode="physical_mm", weighting_mode="physical_mm", sigma_mm=3)
    assert configured["preprocess"]["target_spacing_mm"] == 1.5
    assert configured["loss"]["sigma_mm"] == 3
    assert configured["model"]["presence_hidden"] == 8
    assert configured["loss"]["presence_weight"] == .2
    result = subprocess.run([sys.executable, "scripts/merit_pipeline.py", "check-split", "--config", str(path),
                             "--target-spacing-mm", "1.5"], check=True, capture_output=True, text=True)
    assert len(json.loads(result.stdout)["fold_counts"]) == 5


def test_alignment_max_is_more_sensitive_to_one_high_pixel_than_top_k_mean():
    raw = torch.zeros(1, 2, 3, 4, 4)
    raw[:, 1] = math.log(.1/.9)
    raw[:, 1, :, 0, 0] = math.log(.99/.01)
    raw.requires_grad_()
    output = MERITOutput(torch.zeros_like(raw), torch.full((1,3), .5), None, raw)
    target = torch.zeros(1, 1, 3, 4, 4)
    maximum = MERITLoss(alignment_pool="max")(output, target, [1])["alignment"]
    top_k = MERITLoss(alignment_pool="top_k_mean", top_k_fraction=.25)(output, target, [1])["alignment"]
    assert top_k < maximum
    maximum.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[:, :, :, 0, 0].abs().sum() > 0


def test_physical_training_optimizer_and_native_validation(tmp_path):
    config = synthetic_experiment(tmp_path / "data", sampling_mode="physical_mm", target_spacing_mm=1.5)
    config["model"].update(position_mode="physical_mm", position_unit_mm=1.0)
    config["loss"].update(weighting_mode="physical_mm", sigma_mm=3.0)
    config = validate_experiment(config)
    datasets, info = split_datasets(config, 0)
    loader = make_dataloader(datasets["train"], batch_size=2, shuffle=True, seed=7)
    trainer = Trainer(config, loader, datasets["validation"].cases, info, tmp_path / "run")
    history = trainer.fit()
    assert history[0]["train"]["optimizer_steps"] > 0
    assert history[0]["validation"]["evaluation_grid"] == "full original NIfTI XYZ"
    loaded, cp = load_checkpoint_model(tmp_path / "run" / "last.pt")
    assert loaded.position_mode == "physical_mm"
    assert cp["config"]["loss"]["weighting_mode"] == "physical_mm"
