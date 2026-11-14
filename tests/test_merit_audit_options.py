"""Regression tests for opt-in COVID research controls."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from merit_covid.data.pipeline.dataloader import VolumeBalancedSampler, make_dataloader
from merit_covid.data.pipeline.slab import sample_arrays
from merit_covid.engine.training import augment_slab_flips
from merit_covid.models.merit_unetrpp.modules import MasterEvidenceRouting, TrajectoryEvidencePrototype
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITUNETRPPAdapter, MERITLoss


def test_duplicate_mask_respects_master_and_is_independent_of_loss():
    meta = {"canonical_shape_xyz": [2, 2, 2], "canonical_affine_mm": np.diag([1., 1., 5., 1.])}
    sample = sample_arrays(np.zeros((2, 2, 2)), None, 0, 11, meta, duplicate_slice_policy="all")
    assert sample["slice_valid"].all()
    assert sample["attention_valid"].sum() == 2
    assert sample["attention_valid"][5]
    assert np.array_equal(sample["z_offsets_mm"], sample["slice_indices"] * 5)


def test_masked_tep_and_mer_ignore_context_and_have_finite_gradients():
    torch.manual_seed(12)
    x = torch.randn(2, 4, 3, 2, 2, requires_grad=True)
    valid = torch.tensor([[False, True, True], [False, True, False]])
    tep = TrajectoryEvidencePrototype(4, salience_hidden=3, position_dim=4, position_hidden=3)
    mer = MasterEvidenceRouting(4, attention_dim=4, gate_hidden=3, query_chunk_size=2)
    out, _ = tep(x, [1, 1], attention_valid=valid)
    changed = x.detach().clone()
    changed[0, :, 0] += 100
    changed[1, :, (0, 2)] += 100
    out2, _ = tep(changed, [1, 1], attention_valid=valid)
    torch.testing.assert_close(out[:, :, 1], out2[:, :, 1])
    routed = mer(x, [1, 1], attention_valid=valid)
    routed2 = mer(changed, [1, 1], attention_valid=valid)
    torch.testing.assert_close(routed[:, :, 1], routed2[:, :, 1])
    torch.testing.assert_close(routed[1, :, 1], x[1, :, 1])  # no valid context
    torch.testing.assert_close(routed[:, :, 0], x[:, :, 0])  # only master changes
    (out.square().mean() + routed.square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in list(tep.parameters()) + list(mer.parameters()))
    with pytest.raises(ValueError, match="master"):
        mer(x, [1, 1], attention_valid=torch.zeros(2, 3, dtype=torch.bool))


def test_mer_masked_chunking_matches_full():
    torch.manual_seed(3)
    x = torch.randn(2, 4, 3, 3, 3)
    a = MasterEvidenceRouting(4, attention_dim=4, gate_hidden=3, query_chunk_size=2)
    b = MasterEvidenceRouting(4, attention_dim=4, gate_hidden=3, query_chunk_size=100)
    b.load_state_dict(a.state_dict())
    valid = torch.tensor([[True, True, False], [False, True, False]])
    torch.testing.assert_close(a(x, [1, 1], attention_valid=valid), b(x, [1, 1], attention_valid=valid))


def test_flips_are_seeded_and_shared_across_slices_and_masks():
    x = torch.arange(24).reshape(2, 1, 3, 2, 2)
    batch = {"image_slab": x, "mask_slab": x.clone(), "master_mask": x[:, :, 1].clone(),
             "valid_region": torch.ones_like(x, dtype=torch.bool)}
    torch.manual_seed(5)
    out = augment_slab_flips(batch)
    torch.manual_seed(5)
    again = augment_slab_flips(batch)
    assert torch.equal(out["image_slab"], out["mask_slab"])
    assert torch.equal(out["master_mask"], out["image_slab"][:, :, 1])
    assert torch.equal(out["image_slab"], again["image_slab"])
    assert torch.equal(batch["image_slab"], x)


def test_balanced_sampler_counts_seed_and_train_only():
    dataset = SimpleNamespace(ends=[0, 2, 7], cases=[1, 2], partition="train")
    def draw():
        return list(VolumeBalancedSampler(dataset, 10, torch.Generator().manual_seed(9)))
    a = draw()
    assert a == draw() and sum(i < 2 for i in a) == 10 and sum(i >= 2 for i in a) == 10
    dataset.partition = "validation"
    with pytest.raises(ValueError, match="train-only"):
        make_dataloader(dataset, samples_per_volume=2)


def test_enabled_mask_reaches_training_and_inference_model():
    model = MERITUNETRPP(img_size=(3, 32, 32), feature_size=2, depths=(1, 1, 1, 1),
                         mask_duplicate_context=True)
    image = torch.randn(1, 1, 3, 32, 32)
    valid = torch.tensor([[False, True, False]])
    with pytest.raises(ValueError, match="attention_valid"):
        model(image, [1])
    output = model(image, [1], attention_valid=valid)
    loss = MERITLoss()(output, torch.zeros_like(image), [1])["total"]
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        prediction = MERITUNETRPPAdapter(model)({"image_slab": image, "attention_valid": valid})
    assert prediction.shape == (1, 1, 32, 32) and torch.isfinite(prediction).all()
