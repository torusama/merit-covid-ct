"""Full MER output and parameter gradients against original distance bias."""
from copy import deepcopy
import types

import pytest
import torch

from merit_covid.engine.activation_checkpointing import checkpointed_mer
from merit_covid.models.merit_unetrpp import MasterEvidenceRouting
from tools.merit_factorized_attention import factorized_mer


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("position_mode", ["index", "physical_mm"])
def test_factorized_mer_matches_original_output_and_gradients(masked, training, position_mode):
    torch.manual_seed(813)
    reference = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3, query_chunk_size=5,
                                      position_mode=position_mode)
    candidate = deepcopy(reference)
    reference.forward = types.MethodType(checkpointed_mer, reference)
    candidate.forward = types.MethodType(factorized_mer, candidate)
    reference.train(training)
    candidate.train(training)
    image = torch.randn(2, 4, 3, 4, 6)
    offsets = torch.tensor([[0., 2.5, 7.], [-8., -3., 0.]])
    mask = torch.tensor([[True, True, False], [False, False, True]]) if masked else None
    tensors, outputs = [], []
    for model in (reference, candidate):
        x = image.clone().requires_grad_()
        y = model(x, [0, 2], offsets, mask)
        y.square().mean().backward()
        tensors.append(x)
        outputs.append(y)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tensors[0].grad, tensors[1].grad, rtol=1e-5, atol=1e-7)
    for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-7, msg=name)
    if masked:
        torch.testing.assert_close(outputs[1][1, :, 2], image[1, :, 2], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_factorized_mer_cuda_matches_original_output_and_gradients(masked, amp):
    torch.manual_seed(813)
    reference = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3, query_chunk_size=5,
                                      position_mode="physical_mm").cuda().train()
    candidate = deepcopy(reference)
    reference.forward = types.MethodType(checkpointed_mer, reference)
    candidate.forward = types.MethodType(factorized_mer, candidate)
    image = torch.randn(2, 4, 3, 4, 6).cuda()
    offsets = torch.tensor([[0., 2.5, 7.], [-8., -3., 0.]])
    mask = torch.tensor([[True, True, False], [False, False, True]]) if masked else None
    tensors, outputs = [], []
    for model in (reference, candidate):
        x = image.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            y = model(x, [0, 2], offsets, mask)
            loss = y.square().mean()
        loss.backward()
        tensors.append(x)
        outputs.append(y)
    rtol, atol = (5e-3, 2e-6) if amp else (1e-4, 1e-7)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=rtol, atol=atol)
    torch.testing.assert_close(tensors[0].grad, tensors[1].grad, rtol=rtol, atol=atol)
    for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert a.grad is not None and b.grad is not None
        assert torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad, b.grad, rtol=rtol, atol=atol, msg=name)
    assert candidate.distance_scale_raw.grad.abs() > 0
    if masked:
        torch.testing.assert_close(outputs[1][1, :, 2], image[1, :, 2], rtol=0, atol=0)
