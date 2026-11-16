"""CUDA backend eligibility and gradients against the legacy 3D math path."""
from copy import deepcopy
import types

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from merit_covid.engine import activation_checkpointing as execution
from merit_covid.engine.activation_checkpointing import checkpointed_mer
from merit_covid.models.merit_unetrpp import MasterEvidenceRouting
from merit_covid.models.merit_unetrpp import modules


def _forward_backward(model, image, masters, offsets, mask, *, backend, amp):
    # Checkpoint recomputes inside backward. Keep the selected SDPA backend
    # active there too; autocast itself can end after the original forward.
    with sdpa_kernel(backend):
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=amp):
            result = model(image, masters, offsets, mask)
            loss = result.square().mean()
        loss.backward()
    return result


def test_checkpoint_recompute_keeps_attention_backend_scope_on_cpu(monkeypatch):
    original = F.scaled_dot_product_attention
    observed = []

    def attention(*args, **kwargs):
        observed.append((torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                         torch.backends.cuda.mem_efficient_sdp_enabled()))
        return original(*args, **kwargs)

    def model(image, *unused):
        return checkpoint(lambda x: F.scaled_dot_product_attention(x, x, x, dropout_p=0.0),
                          image, use_reentrant=False)

    monkeypatch.setattr(F, "scaled_dot_product_attention", attention)
    image = torch.randn(1, 1, 4, 8, requires_grad=True)
    _forward_backward(model, image, None, None, None, backend=SDPBackend.MATH, amp=False)
    assert len(observed) >= 2  # Original forward and checkpoint recomputation.
    assert all(state == (True, False, False) for state in observed)
    assert torch.isfinite(image.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_mer_efficient_backend_matches_legacy_math_gradients(monkeypatch, recompute, masked, amp):
    torch.manual_seed(93)
    reference = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3,
        query_chunk_size=5, position_mode="physical_mm").cuda().train()
    candidate = deepcopy(reference)
    if recompute:
        reference.forward = types.MethodType(checkpointed_mer, reference)
        candidate.forward = types.MethodType(checkpointed_mer, candidate)
    inputs = torch.randn(2, 4, 3, 4, 6).cuda()
    offsets = torch.tensor([[0., 2.5, 7.], [-8., -3., 0.]])
    mask = torch.tensor([[True, True, False], [False, False, True]]) if masked else None
    def legacy_math(query, key, value, bias):
        return F.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=0.0)

    x = inputs.clone().requires_grad_()
    with monkeypatch.context() as patch:
        patch.setattr(modules, "_mer_attention", legacy_math, raising=False)
        patch.setattr(execution, "_mer_attention", legacy_math, raising=False)
        expected = _forward_backward(reference, x, [0, 2], offsets, mask, backend=SDPBackend.MATH, amp=amp)

    y = inputs.clone().requires_grad_()
    # No math fallback: exercise fused eligibility during forward AND recompute.
    actual = _forward_backward(candidate, y, [0, 2], offsets, mask,
                               backend=SDPBackend.EFFICIENT_ATTENTION, amp=amp)
    rtol, atol = (5e-3, 2e-6) if amp else (1e-4, 1e-7)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(y.grad, x.grad, rtol=rtol, atol=atol)
    for (name, left), (other, right) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert name == other and left.grad is not None and right.grad is not None
        assert torch.isfinite(right.grad).all(), name
        torch.testing.assert_close(right.grad, left.grad, rtol=rtol, atol=atol, msg=name)
    assert candidate.distance_scale_raw.grad.abs() > 0
    if masked:
        # The second sample has no valid context; its master must be unchanged.
        torch.testing.assert_close(actual[1, :, 2], inputs[1, :, 2], rtol=0, atol=0)
