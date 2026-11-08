"""UNETR++ software checks, exclusively synthetic; never update weights."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("monai", reason='Install the model extra: pip install -e ".[unetrpp]"')
pytest.importorskip("einops", reason='Install the model extra: pip install -e ".[unetrpp]"')

from merit_covid import PreprocessConfig
from merit_covid.data.pipeline.cases import Case
from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.models.unetrpp import UNETRPP, UNETRPPAdapter
from merit_covid.models.unetrpp.decoder import UnetrUpBlock
from merit_covid.models.unetrpp.epa import EPA


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def small_model(**kwargs):
    return UNETRPP(img_size=(32, 32, 32), feature_size=2, **kwargs)


def test_epa_matches_separate_equation_reference_and_shared_projection():
    """Independent branch contractions catch changed scaling or head layouts."""
    torch.manual_seed(7)
    epa = EPA(12, 8, 5, num_heads=2, channel_attn_drop=0, spatial_attn_drop=0).double()
    x = torch.randn(2, 12, 8, dtype=torch.float64, requires_grad=True)
    assert epa.E is epa.F and epa.E.weight.data_ptr() == epa.F.weight.data_ptr()
    q, k, vc, vs = epa.qkvv(x).reshape(2, 12, 4, 2, 4).permute(2, 0, 3, 4, 1).unbind(0)
    kp, vp = epa.E(k), epa.F(vs)
    q, k = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1)
    channel_weights = (torch.einsum("bhcn,bhkn->bhck", q, k) * epa.temperature).softmax(-1)
    channel = torch.einsum("bhck,bhkn->bhcn", channel_weights, vc).permute(0, 3, 1, 2).reshape(2, 12, 8)
    spatial_weights = (torch.einsum("bhcn,bhcp->bhnp", q, kp) * epa.temperature2).softmax(-1)
    spatial = torch.einsum("bhnp,bhcp->bhnc", spatial_weights, vp).permute(0, 3, 1, 2).reshape(2, 12, 8)
    expected = torch.cat([epa.out_proj(spatial), epa.out_proj2(channel)], dim=-1)
    actual = epa(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in epa.parameters())


def test_original_configuration_forward_encoder_decoder_and_head_shapes():
    config = Path(__file__).resolve().parents[1] / "configs/unetrpp_baseline.yaml"
    model = UNETRPP(**yaml.safe_load(config.read_text(encoding="utf-8"))["model"]).eval()
    assert sum(p.numel() for p in model.parameters()) == 42_951_355
    assert model.stage_shapes == ((32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4))
    assert [len(stage) for stage in model.unetr_pp_encoder.stages] == [3, 3, 3, 3]
    shapes = {}
    hooks = [module.register_forward_hook(
        lambda _m, _args, output, name=name: shapes.__setitem__(name, tuple(output.shape))
    ) for name, module in (("decoder5", model.decoder5), ("decoder4", model.decoder4),
                          ("decoder3", model.decoder3), ("decoder2", model.decoder2))]
    with torch.inference_mode():
        result = model(torch.randn(1, 1, 64, 128, 128))
    for hook in hooks:
        hook.remove()
    assert shapes == {"decoder5": (1, 128, 8, 8, 8), "decoder4": (1, 64, 16, 16, 16),
                      "decoder3": (1, 32, 32, 32, 32), "decoder2": (1, 16, 64, 128, 128)}
    assert [tuple(t.shape) for t in result] == [(1, 1, 64, 128, 128), (1, 1, 32, 32, 32), (1, 1, 16, 16, 16)]
    assert all(torch.isfinite(t).all() for t in result)
    assert not any(isinstance(m, (torch.nn.Sigmoid, torch.nn.Softmax)) for m in model.modules())


@pytest.mark.parametrize("do_ds", [False, True])
def test_compact_backward_all_parameters_without_weight_update(do_ds):
    model = small_model(do_ds=do_ds)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    image = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
    output = model(image)
    assert isinstance(output, list) is do_ds
    logits = output if do_ds else [output]
    sum(t.square().mean() for t in logits).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.equal(before[name], parameter), name


def test_non_cubic_encoder_and_multichannel_configuration():
    model = UNETRPP(in_channels=2, out_channels=3, img_size=(16, 32, 64), feature_size=2, do_ds=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 2, 16, 32, 64))
    assert output.shape == (1, 3, 16, 32, 64)
    assert model.unetr_pp_encoder.downsample_layers[0][0].conv.in_channels == 2


def test_decoder_uses_addition_and_checks_skip_shape():
    decoder = UnetrUpBlock(3, 8, 4, 3, 2, "instance", conv_decoder=True).eval()
    image, skip = torch.randn(1, 8, 2, 2, 2), torch.randn(1, 4, 4, 4, 4)
    with torch.no_grad():
        expected = decoder.decoder_block[0](decoder.transp_conv(image) + skip)
        torch.testing.assert_close(decoder(image, skip), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="skip shape"):
        decoder(image, skip[:, :, :-1])


@pytest.mark.parametrize("do_ds", [False, True])
def test_adapter_two_batch_contracts_and_distinct_master_meanings(do_ds):
    model = small_model(do_ds=do_ds).eval()
    adapter = UNETRPPAdapter(model)
    image = torch.randn(2, 1, 32, 32, 32)
    image_before = image.clone()
    with torch.no_grad():
        raw = model(image)
        primary = raw[0] if do_ds else raw
        batch = {"image_slab": image, "master_slice_index": torch.tensor([100, 200]),
                 "master_mask": object(), "metadata": object(), "z_offsets_mm": object()}
        actual = adapter.predict_logits(batch)
        torch.testing.assert_close(actual, primary[:, :, 16], rtol=0, atol=0)
        actual = adapter.predict_logits({"image": image, "master_index": torch.tensor([0, 31]),
                                         "slice_index": [100, 200]})
        torch.testing.assert_close(actual, primary[torch.arange(2), :, torch.tensor([0, 31])], rtol=0, atol=0)
    assert adapter.input_kind == "slab_3d" and torch.equal(image, image_before)


def test_adapter_backward_retains_gradients_with_master_only_target():
    adapter = UNETRPPAdapter(small_model(do_ds=False))
    image = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
    master_target = torch.zeros(1, 1, 32, 32)
    logits = adapter.predict_logits({"image_slab": image, "master_slice_index": [123]})
    torch.nn.functional.binary_cross_entropy_with_logits(logits, master_target).backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert adapter.model.out1.conv.conv.weight.grad is not None


def test_current_pipeline_batch_is_rejected_without_any_implicit_transform(nifti_case):
    case, _, _ = nifti_case(shape=(7, 9, 4), affine=np.diag([-1.0, -1.0, 5.0, 1.0]))
    dataset = CTSlabDataset([Case(case.case_id, case.image, case.infection_mask, case.case_id)], PreprocessConfig())
    batch = next(iter(make_dataloader(dataset, batch_size=2)))
    assert batch["image_slab"].shape == (2, 1, 11, 9, 7)
    original = batch["image_slab"].clone()
    with pytest.raises(ValueError, match="not directly supported"):
        UNETRPPAdapter(small_model()).predict_logits(batch)
    assert torch.equal(original, batch["image_slab"])


@pytest.mark.parametrize("kwargs", [
    {"img_size": (11, 512, 512)}, {"img_size": (16, 32, 32)},
    {"img_size": (32, 31, 32)}, {"img_size": (32, 32)}, {"img_size": "32,32,32"},
    {"img_size": (True, 32, 32)}, {"feature_size": 0}, {"feature_size": 3},
    {"depths": (3, 3, 3)}, {"depths": (0, 3, 3, 3)}, {"hidden_size": 128},
    {"dims": (32, 64, 128, 128)}, {"in_channels": 3}, {"out_channels": 0},
    {"num_heads": 3}, {"num_heads": 0}, {"do_ds": "true"},
])
def test_invalid_model_configuration(kwargs):
    with pytest.raises(ValueError):
        UNETRPP(**kwargs)


@pytest.mark.parametrize("shape,dtype", [
    ((1, 1, 11, 512, 512), torch.float32), ((1, 1, 16, 32, 32), torch.float32),
    ((1, 2, 32, 32, 32), torch.float32), ((0, 1, 32, 32, 32), torch.float32),
    ((1, 32, 32, 32), torch.float32), ((1, 1, 32, 32, 32), torch.int16),
])
def test_invalid_model_input(shape, dtype):
    with pytest.raises(ValueError):
        small_model()(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("indices", [[-1], [32], [1.5], [True], [], [0, 1],
                                      torch.tensor([2.0]), torch.tensor([True]), torch.tensor([[2]])])
def test_invalid_adapter_master_index(indices):
    with pytest.raises(ValueError):
        UNETRPPAdapter(small_model()).predict_logits({"image": torch.zeros(1, 1, 32, 32, 32), "master_index": indices})


def test_missing_or_ambiguous_adapter_fields():
    model = small_model()
    adapter = UNETRPPAdapter(model)
    image = torch.zeros(1, 1, 32, 32, 32)
    for batch in ({}, {"image": image}, {"image": image, "image_slab": image}):
        with pytest.raises(ValueError):
            adapter.predict_logits(batch)
    with pytest.raises(ValueError, match="out_channels=1"):
        UNETRPPAdapter(small_model(out_channels=2))


@pytest.mark.parametrize("kwargs", [
    {"input_size": 0}, {"hidden_size": 7}, {"num_heads": 3},
    {"channel_attn_drop": 1.1}, {"spatial_attn_drop": -0.1},
])
def test_invalid_epa_configuration(kwargs):
    config = dict(input_size=8, hidden_size=8, proj_size=4, num_heads=2)
    with pytest.raises(ValueError):
        EPA(**(config | kwargs))


@pytest.mark.parametrize("shape", [(2, 7, 8), (2, 8, 6), (2, 8), (0, 8, 8)])
def test_invalid_epa_input(shape):
    with pytest.raises(ValueError):
        EPA(8, 8, 4, 2)(torch.randn(*shape))


def test_optional_model_dependencies_are_not_imported_by_numpy_pipeline():
    source = """
import sys
sys.modules['torch'] = None
sys.modules['monai'] = None
sys.modules['einops'] = None
import merit_covid
from merit_covid.data.nifti_io import load_case
from merit_covid.models.base import master_slice_input
assert 'merit_covid.models.unetrpp' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)
