"""Chunking retains the same full-context MER outputs and gradients."""
from copy import deepcopy
import types

import pytest
import torch

from merit_covid.models.merit_unetrpp import MasterEvidenceRouting
from tools.merit_factorized_attention import factorized_mer


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_factorized_query_chunks_match(masked, amp):
    if amp and not torch.cuda.is_available():
        pytest.skip("AMP requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(812)
    reference = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3,
                                      query_chunk_size=7, position_mode="physical_mm").to(device).train()
    candidate = deepcopy(reference)
    for model in (reference, candidate):
        model.forward = types.MethodType(factorized_mer, model)
    candidate.query_chunk_size = 13  # includes a partial final chunk
    image = torch.randn(2, 4, 3, 5, 6, device=device)
    offsets = torch.tensor([[0., 2.5, 7.], [-8., -3., 0.]])
    mask = torch.tensor([[True, True, False], [False, False, True]]) if masked else None
    results = []
    for model in (reference, candidate):
        x = image.detach().clone().requires_grad_()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(x, [0, 2], offsets, mask)
            loss = output.square().mean()
        loss.backward()
        results.append((output.detach(), x.grad.detach(),
                        {name: p.grad.detach().clone() for name, p in model.named_parameters()}))
    rtol, atol = (5e-3, 2e-6) if amp else (1e-4, 1e-7)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=rtol, atol=atol)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=rtol, atol=atol)
    assert results[0][2].keys() == results[1][2].keys()
    for name in results[0][2]:
        assert torch.isfinite(results[1][2][name]).all(), name
        torch.testing.assert_close(results[0][2][name], results[1][2][name], rtol=rtol, atol=atol, msg=name)
