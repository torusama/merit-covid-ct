import pytest
import torch
from merit_covid.models.merit_unetrpp.loss import probability_bce


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_bce_input_is_rejected_without_poisoning_device(device, value):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    bad = torch.tensor([value], device=device)
    target = torch.zeros_like(bad)
    with pytest.raises((FloatingPointError, ValueError), match='before BCE'):
        probability_bce(bad, target, name='presence')
    good = torch.tensor([0.7], device=device, requires_grad=True)
    probability_bce(good, target).sum().backward()
    assert torch.isfinite(good.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('kind', [torch.nn.Conv3d, torch.nn.ConvTranspose3d])
@pytest.mark.parametrize('recompute', [False, True])
def test_amp_convolution_overflow_retries_fp32_with_correct_gradients(kind, recompute):
    from merit_covid.engine.numerical_safety import enable_safe_amp_convolutions
    conv = kind(1,1,1,bias=False).cuda()
    conv.weight.data.fill_(2)
    image = torch.full((1,1,1,2,2),40000.,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.float16):
        broken = conv(image)
    assert not torch.isfinite(broken).all()
    enable_safe_amp_convolutions(conv)
    with torch.autocast('cuda',dtype=torch.float16):
        from torch.utils.checkpoint import checkpoint
        actual = checkpoint(conv,image,use_reentrant=False) if recompute else conv(image)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual,torch.full_like(actual,80000.))
    actual.sum().backward()
    torch.testing.assert_close(image.grad,torch.full_like(image,2.))
    torch.testing.assert_close(conv.weight.grad,torch.full_like(conv.weight,160000.))
    assert conv._fp32_retry_count >= 1
