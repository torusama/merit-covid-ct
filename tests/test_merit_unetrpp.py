"""Paper-equation, integration and gradient tests; no learned-quality claims."""

from pathlib import Path
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import yaml

pytest.importorskip("monai")
pytest.importorskip("einops")

from merit_covid.models.merit_unetrpp import (
    MERITLoss, MERITOutput, MERITUNETRPP, MERITUNETRPPAdapter,
    MasterEvidenceRouting, PresenceCommitmentGate, TrajectoryEvidencePrototype,
)
from merit_covid.models.merit_unetrpp.modules import sinusoidal_encoding


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def small_model(**kwargs):
    return MERITUNETRPP(img_size=(11, 32, 32), feature_size=2, depths=(1, 1, 1, 1), **kwargs)


def reference_pe(offsets, channels):
    result = torch.empty(*offsets.shape, channels, dtype=torch.float64)
    for channel in range(channels):
        angle = offsets.double() / 10000 ** (2 * (channel // 2) / channels)
        result[..., channel] = angle.sin() if channel % 2 == 0 else angle.cos()
    return result


def test_paper_table3_module_parameter_counts():
    counts = [sum(p.numel() for p in module.parameters()) for module in (
        TrajectoryEvidencePrototype(), MasterEvidenceRouting(), PresenceCommitmentGate())]
    assert counts == [279_731, 9_252, 145]
    assert sum(counts) == 289_128


def test_signed_position_encoding_matches_independent_formula():
    offsets = torch.tensor([[-2, -1, 0, 1, 2]])
    torch.testing.assert_close(sinusoidal_encoding(offsets, 8, dtype=torch.float64), reference_pe(offsets, 8))
    assert not torch.equal(sinusoidal_encoding(offsets, 8)[0, 1], sinusoidal_encoding(offsets, 8)[0, 3])


def test_tep_matches_equations_and_salience_is_content_only():
    torch.manual_seed(12)
    module = TrajectoryEvidencePrototype(4, salience_hidden=3, position_dim=4, position_hidden=2).double()
    features = torch.randn(2, 4, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    master = torch.tensor([0, 2])
    slices = features.permute(0, 2, 3, 4, 1).reshape(2, 3, 4, 4)
    salience = module.salience_head(slices.mean(2)).squeeze(-1).float().sigmoid()
    pe = reference_pe(torch.arange(3)[None] - master[:, None], 4)
    scores = module.position_head(torch.cat((salience[..., None].double(), pe), -1)).squeeze(-1)
    weights = (scores.float() / module.temperature.float()).softmax(-1).double()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=torch.float64))
    prototype = torch.einsum("bz,bznc->bnc", weights, slices)
    q, k, v = module.query(slices), module.key(prototype), module.value(prototype)
    attention = (torch.einsum("bznc,bkc->bznk", q, k) / 2).softmax(-1)
    expected = slices + module.output(torch.einsum("bznk,bkc->bznc", attention, v))
    result, actual_salience = module(features, master)
    torch.testing.assert_close(result.permute(0, 2, 3, 4, 1).reshape_as(expected), expected)
    torch.testing.assert_close(actual_salience, salience)
    other, other_salience = module(features, [2, 0])
    torch.testing.assert_close(other_salience, actual_salience, rtol=0, atol=0)
    assert not torch.allclose(other, result, rtol=0, atol=1e-10)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_mer_matches_equations_context_exclusion_and_chunked_gradients():
    torch.manual_seed(5)
    module = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3, query_chunk_size=2).double()
    features = torch.randn(2, 4, 3, 2, 3, dtype=torch.float64, requires_grad=True)
    masters = [0, 2]
    expected = features.clone()
    coords = torch.tensor([(i, j) for i in range(2) for j in range(3)])
    distances = (coords[:, None] - coords[None]).square().sum(-1) / 13.0
    for b, m in enumerate(masters):
        fm = features[b, :, m].flatten(1).T
        contexts = torch.stack([features[b, :, z].flatten(1).T for z in range(3) if z != m])
        offsets = torch.tensor([z - m for z in range(3) if z != m])
        pe = reference_pe(offsets, 4)[:, None]
        q = module.query(fm)
        k = module.key(contexts + module.position_scale * pe).flatten(0, 1)
        v = module.value(contexts).flatten(0, 1)
        weights = (q @ k.T / math.sqrt(8) - module.distance_scale * distances.repeat(1, 2)).softmax(-1)
        update = module.output(weights @ v)
        gate = module.gate(torch.cat((fm, update), -1))
        expected[b, :, m] = (fm + module.beta * gate * update).T.reshape(4, 2, 3)
    actual = module(features, masters)
    torch.testing.assert_close(actual, expected)
    for b, m in enumerate(masters):
        for z in range(3):
            if z != m:
                assert torch.equal(actual[b, :, z], features[b, :, z])
    gradient = torch.autograd.grad(actual.square().sum(), features, retain_graph=True)[0]
    module.query_chunk_size = 100
    unchunked = module(features, masters)
    torch.testing.assert_close(actual, unchunked)
    torch.testing.assert_close(gradient, torch.autograd.grad(unchunked.square().sum(), features)[0])
    actual.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("probability", [0.2, 0.8])
def test_pcg_foreground_log_odds_background_unchanged_and_warmup(probability):
    gate = PresenceCommitmentGate(4, hidden=2)
    with torch.no_grad():
        gate.presence_head[-1].weight.zero_()
        gate.presence_head[-1].bias.fill_(math.log(probability / (1 - probability)))
    features = torch.randn(2, 4, 3, 2, 2)
    raw = torch.randn(2, 2, 3, 2, 2)
    saved = raw.clone()
    final, presence = gate(features, raw, ramp=0.5)
    torch.testing.assert_close(presence, torch.full((2, 3), probability))
    assert torch.equal(final[:, 0], raw[:, 0]) and torch.equal(raw, saved)
    torch.testing.assert_close(final[:, 1] - raw[:, 1], torch.full((2, 3, 2, 2), 0.5 * math.log(probability / (1 - probability))))
    assert torch.equal(gate(features, raw, ramp=0)[0], raw)
    with torch.no_grad():
        gate.presence_head[-1].bias.fill_(1000)
    assert torch.isfinite(gate(features, raw)[0]).all()


def test_default_paper_grid_and_insertion_shapes():
    path = Path(__file__).resolve().parents[1] / "configs/merit_unetrpp.yaml"
    model = MERITUNETRPP(**yaml.safe_load(path.read_text())["model"]).eval()
    assert model.stage_shapes == ((11, 32, 32), (11, 16, 16), (11, 8, 8), (11, 4, 4))
    observed = {}
    hooks = [module.register_forward_pre_hook(
        lambda _m, args, name=name: observed.__setitem__(name, tuple(args[0].shape)))
        for name, module in (("tep", model.tep), ("mer", model.mer), ("pcg", model.pcg))]
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 11, 128, 128), [9])
    for hook in hooks:
        hook.remove()
    assert observed == {"tep": (1, 256, 11, 4, 4), "mer": (1, 32, 11, 32, 32), "pcg": (1, 16, 11, 128, 128)}
    assert output.logits.shape == output.raw_logits.shape == (1, 2, 11, 128, 128)
    assert output.presence.shape == output.salience.shape == (1, 11)
    assert all(torch.isfinite(t).all() for t in output)


@pytest.mark.parametrize("flags", [(False, False, False), (True, False, False), (False, True, False),
                                  (True, True, False), (True, False, True), (True, True, True),
                                  (False, False, True), (False, True, True)])
def test_ablation_losses_backward_and_checkpoint_ramp(flags):
    model = small_model(**dict(zip(("use_tep", "use_mer", "use_pcg"), flags)))
    model.set_epoch(15)
    assert model.pcg_ramp.item() == 0.5
    image = torch.randn(2, 1, 11, 32, 32, requires_grad=True)
    target = torch.zeros(2, 1, 11, 32, 32)
    target[0, :, 2, 5:10, 6:11] = 1
    target[1, :, 8, 15:20, 15:20] = 1
    result = model(image, [2, 8])
    assert (result.salience is not None) == flags[0]
    assert (result.presence is not None) == flags[2]
    loss = MERITLoss()(result, target, [2, 8])
    assert all(torch.isfinite(value) for value in loss.values())
    loss["total"].backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored = small_model(**dict(zip(("use_tep", "use_mer", "use_pcg"), flags)))
    restored.load_state_dict(model.state_dict())
    assert restored.pcg_ramp.item() == 0.5
    model.set_epoch(100)
    assert model.pcg_ramp.item() == 1


def test_loss_gaussian_pair_reduction_raw_alignment_and_padding():
    logits = torch.tensor([[[[[0.0]], [[1.0]], [[2.0]]], [[[1.0]], [[0.0]], [[-1.0]]]]], requires_grad=True)
    raw = torch.zeros_like(logits, requires_grad=True)
    target = torch.tensor([[[[1]], [[0]], [[0]]]])
    presence = torch.tensor([[0.8, 0.2, 0.4]], requires_grad=True)
    salience = torch.tensor([[0.7, 0.3, 0.5]], requires_grad=True)
    criterion = MERITLoss(gaussian_sigma=1.0, presence_positive_weight=2.0)
    result = criterion(MERITOutput(logits, presence, salience, raw), target, [0])
    weights = torch.tensor([1.0, math.exp(-0.5), math.exp(-2.0)])
    probabilities = logits.softmax(1)[0, 1, :, 0, 0]
    truth = target[0, :, 0, 0].float()
    dice = 1 - (2 * probabilities * truth + criterion.smooth) / (probabilities + truth + criterion.smooth)
    ce = F.cross_entropy(logits, target, reduction="none").flatten()
    torch.testing.assert_close(result["segmentation"], ((dice + ce) * weights).sum() / weights.sum())
    bce = F.binary_cross_entropy(presence[0], truth, reduction="none")
    torch.testing.assert_close(result["presence"], (bce * torch.tensor([2., 1., 1.]) * weights).sum() / weights.sum())
    torch.testing.assert_close(result["alignment"], ((0.5 - presence).square()).mean())
    result["total"].backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, 2] = False
    target[:, 2] = 999  # Invalid/padded labels must never be used as GT.
    original = criterion(MERITOutput(logits, presence, salience, raw), target, [0], valid_region=valid)
    changed_logits = logits.detach().clone()
    changed_logits[:, :, 2] = 100
    changed = criterion(MERITOutput(changed_logits, presence, salience, raw), target, [0], valid_region=valid)
    torch.testing.assert_close(original["total"], changed["total"])
    with pytest.raises(ValueError, match="actual full-slab"):
        criterion(MERITOutput(logits, presence, salience, raw), torch.zeros(1, 1, 1, 1), [0])


def test_adapter_softmax_equivalence_distinct_masters_and_no_gt_reads():
    model = small_model().eval()
    adapter = MERITUNETRPPAdapter(model)
    image = torch.randn(2, 1, 11, 32, 32)
    with torch.no_grad():
        output = model(image, [0, 10])
        actual = adapter.predict_logits({"image": image, "master_index": [0, 10],
                                         "mask": object(), "master_slice_index": [100, 200]})
        expected = output.logits.softmax(1)[torch.arange(2), 1, torch.tensor([0, 10])]
        torch.testing.assert_close(actual.sigmoid()[:, 0], expected)
        centered = adapter({"image_slab": image, "master_slice_index": [100, 200], "master_mask": object()})
        explicit = adapter({"image": image, "master_index": [5, 5]})
        torch.testing.assert_close(centered, explicit)
    with pytest.raises(ValueError, match="explicit master_index"):
        adapter({"image": image})
    with pytest.raises(ValueError, match="exactly one"):
        adapter({"image": image, "image_slab": image})


@pytest.mark.parametrize("indices", [[-1], [11], [1.5], [True], [], [0, 1], torch.tensor([2.0])])
def test_invalid_masters_fail_before_encoder(indices):
    model = small_model()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, 11, 32, 32), indices)


def test_real_reader_patch_full_gt_and_valid_mask_integration(nifti_case):
    from merit_covid.data.pipeline.cases import Case
    from merit_covid.data.unetrpp_input import PatchConfig, UNETRPPPatchDataset, make_patch_dataloader
    case, _, _ = nifti_case(shape=(20, 24, 4), affine=np.diag([-1., -1., 2., 1.]))
    dataset = UNETRPPPatchDataset([Case(case.case_id, case.image, case.infection_mask, case.case_id)],
                                PatchConfig(patch_size=(11, 32, 32)), require_mask=True)
    batch = next(iter(make_patch_dataloader(dataset)))
    assert batch["image_patch"].shape == (1, 1, 11, 32, 32)
    model = small_model()
    output = model(batch["image_patch"], [1])
    losses = MERITLoss()(output, batch["mask_patch"], [1], valid_region=batch["valid_region"])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_disabled_auxiliary_losses_do_not_overflow_on_full_grid_fp16():
    logits = torch.ones(1, 2, 11, 128, 128, dtype=torch.float16, requires_grad=True)
    assert torch.isinf(logits.sum())  # The regression trigger is a finite input.
    target = torch.zeros(1, 1, 11, 128, 128)
    losses = MERITLoss()(MERITOutput(logits, None, None, logits), target, [5])
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_cpu_autocast_forward_and_backward():
    model = small_model()
    image = torch.randn(1, 1, 11, 32, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(image, [8])
        losses = MERITLoss()(output, torch.zeros_like(image), [8])
    assert all(torch.isfinite(t).all() for t in output)
    losses["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
