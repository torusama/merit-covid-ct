"""Compare recomputation changes, large chunks, masks and gradients."""
import pytest
import torch

from tools.merit_execution_tuning import compare


@pytest.mark.parametrize("recompute", [True, False])
@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("amp", [True, False])
def test_full_query_and_recomputation(recompute, masked, amp):
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA AMP required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compare(16384, recompute, native=False, amp=amp, masked=masked, device=device)


@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("amp", [True, False])
def test_parallel_reductions_match_reference(masked, amp):
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA AMP required")
    previous = torch.are_deterministic_algorithms_enabled()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compare(4096, True, native=False, amp=amp, masked=masked, device=device, fast_reductions=True)
    assert torch.are_deterministic_algorithms_enabled() == previous


@pytest.mark.parametrize("amp", [True, False])
def test_parallel_reductions_full_model(amp):
    from copy import deepcopy
    from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
    from tools.merit_execution_tuning import deterministic_mode, execution, factor
    if amp and not torch.cuda.is_available():
        pytest.skip("CUDA AMP required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(925)
    model = MERITUNETRPP(img_size=(11,32,32),feature_size=2,depths=(1,1,1,1),query_chunk_size=4096).to(device).train()
    models = [model,deepcopy(model)]
    for m in models:
        factor.enable_factorized_mer(execution.configure_model_execution(m,{"activation_checkpointing":True,"amp":amp}))
    x=torch.randn(1,1,11,32,32,device=device)
    target=torch.zeros_like(x);target[:,:,:,6:9,8:12]=1
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None
    results=[]
    for m,deterministic in zip(models,(True,False)):
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:torch.cuda.set_rng_state_all(gpu_rng)
        xi=x.detach().clone().requires_grad_()
        with deterministic_mode(deterministic):
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                prediction=m(xi,[5])
                losses=MERITLoss()(prediction,target,[5])
            losses["total"].backward()
        results.append((prediction.logits.detach(),xi.grad.detach(),losses))
    rtol,atol=(5e-3,2e-6) if amp else (1e-4,1e-7)
    torch.testing.assert_close(results[0][0],results[1][0],rtol=rtol,atol=atol)
    torch.testing.assert_close(results[0][1],results[1][1],rtol=rtol,atol=atol)
    for key in results[0][2]:
        torch.testing.assert_close(results[0][2][key],results[1][2][key],rtol=rtol,atol=atol,msg=key)
    for (name,p),(_,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
        assert p.grad is not None and q.grad is not None,name
        torch.testing.assert_close(p.grad,q.grad,rtol=rtol,atol=atol,msg=name)
