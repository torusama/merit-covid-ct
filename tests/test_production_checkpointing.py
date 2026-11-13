"""Native input preservation and production recomputation/resume contracts."""

from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.engine.activation_checkpointing import enable_activation_checkpointing
from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import (
    load_experiment, protocol_fingerprint, split_datasets, validate_experiment, validate_native_grid,
)
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
from tools import merit_checkpoint_trial as frozen_trial


def test_production_outputs_all_gradients_rng_and_buffers_match(monkeypatch):
    # Reuse the full-model numerical oracle against the original execution,
    # switching only the overlay under test. It covers train and no-grad eval.
    monkeypatch.setattr(frozen_trial, "enable_checkpointing", enable_activation_checkpointing)
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = frozen_trial.model_equivalence(torch.device("cpu"))
    finally:
        torch.set_num_threads(old)
    assert result["status"] == "PASS"


@pytest.mark.parametrize("mode", ["index", "physical_mm"])
def test_production_matches_frozen_t4_overlay_on_irregular_queries(mode):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(73)
        model = MERITUNETRPP(img_size=(11, 32, 32), feature_size=2, depths=(1, 1, 1, 1),
                            position_mode=mode, query_chunk_size=7).train()
        frozen = frozen_trial.enable_checkpointing(deepcopy(model))
        production = enable_activation_checkpointing(model)
        assert enable_activation_checkpointing(production) is production
        x = torch.randn(2, 1, 11, 32, 32)
        target = torch.zeros_like(x)
        target[:, :, :, 6, 8] = 1
        offsets = (torch.arange(11).float()[None] - torch.tensor([0, 10])[:, None]) * 2.5
        rng = torch.get_rng_state()
        outputs, inputs = [], []
        for m in (frozen, production):
            torch.set_rng_state(rng)
            image = x.detach().clone().requires_grad_()
            out = m(image, [0, 10], offsets)
            loss = MERITLoss()(out, target, [0, 10], z_offsets_mm=offsets)["total"]
            loss.backward()
            outputs.append(loss.detach())
            inputs.append(image.grad)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        torch.testing.assert_close(*inputs, rtol=0, atol=0)
        for (n, p), (other, q) in zip(frozen.named_parameters(), production.named_parameters()):
            assert n == other and p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        for n, p in frozen.state_dict().items():
            torch.testing.assert_close(p, production.state_dict()[n], rtol=0, atol=0)
        frozen.eval()
        production.eval()
        with torch.inference_mode():
            torch.testing.assert_close(frozen(x, [0, 10], offsets).logits,
                                       production(x, [0, 10], offsets).logits, rtol=0, atol=0)
    finally:
        torch.set_num_threads(old)


def test_native_512_config_and_one_pixel_lesion_survive_without_resize(tmp_path, monkeypatch):
    config = load_experiment(Path(__file__).resolve().parents[1] / "configs/merit_native512.yaml")
    assert config["preprocess"]["size_hw"] is None
    assert config["preprocess"]["intensity_encoding"] == "native"
    assert config["preprocess"]["assume_mm"] is False
    assert config["training"]["activation_checkpointing"] is True
    ct = np.full((512, 512, 2), -1234., dtype=np.float32)
    mask = np.zeros(ct.shape, dtype=np.uint8)
    mask[3, 5, 0] = 1
    paths = []
    for name, values in (("ct", ct), ("gt", mask)):
        path = tmp_path / f"{name}.nii.gz"
        image = nib.Nifti1Image(values, np.diag([-1., -1., 5., 1.]))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        paths.append(path)
    case = Case("tiny", *paths, "tiny")
    before = [p.read_bytes() for p in paths]
    preprocess = PreprocessConfig(**config["preprocess"])
    validate_native_grid([case], preprocess, config["model"]["img_size"])
    def forbidden(*args, **kwargs):
        raise AssertionError("native input must not invoke XY interpolation")
    monkeypatch.setattr("merit_covid.data.pipeline.preprocessing.resize_in_plane", forbidden)
    sample = CTSlabDataset([case], preprocess)[0]
    assert sample["image_slab"].shape == (1, 11, 512, 512)
    assert (sample["image_slab"] == -1234.).all()
    assert sample["mask_slab"][0, 5, 5, 3] == 1
    assert not sample["metadata"]["lesion_retention"]["resize_applied"]
    assert before == [p.read_bytes() for p in paths]


def test_native_grid_mismatch_rejected_before_training(tmp_path):
    config = synthetic_experiment(tmp_path / "data")
    config["preprocess"]["size_hw"] = None
    config = validate_experiment(config)
    with pytest.raises(ValueError, match="refusing to resize"):
        split_datasets(config, 0)


def test_legacy_disabled_execution_fingerprint_and_invalid_policy(tmp_path):
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    legacy = deepcopy(config)
    legacy["training"].pop("activation_checkpointing")
    assert protocol_fingerprint(config, {}) == protocol_fingerprint(legacy, {})
    config["training"]["activation_checkpointing"] = "true"
    with pytest.raises(ValueError, match="activation_checkpointing must be boolean"):
        validate_experiment(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_production_cuda_amp_outputs_gradients_and_buffers(monkeypatch):
    monkeypatch.setattr(frozen_trial, "enable_checkpointing", enable_activation_checkpointing)
    result = frozen_trial.model_equivalence(torch.device("cuda"), amp=True)
    assert result["status"] == "PASS" and result["normalization_buffers_unchanged"]
