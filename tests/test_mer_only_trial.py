"""One execution variant; verify equivalence and that backbone regions stay normal."""
import pytest
import torch

from tools import merit_checkpoint_trial as memory
from tools import merit_real_case_trial as pilot
from tools.merit_mer_only_trial import enable_mer_only, candidate_execution
from merit_covid.models.merit_unetrpp import MERITUNETRPP
from merit_covid.models.unetrpp.epa import EPA


def test_mer_only_preserves_backbone_methods_and_state_objects():
    model = MERITUNETRPP(img_size=(11,32,32), feature_size=2, depths=(1,1,1,1))
    before = {n: id(p) for n, p in model.state_dict(keep_vars=True).items()}
    enable_mer_only(model)
    assert before == {n: id(p) for n, p in model.state_dict(keep_vars=True).items()}
    assert all(m.forward.__func__ is EPA.forward for m in model.modules() if isinstance(m, EPA))
    assert "forward" not in model.tep.__dict__
    assert "forward" not in model.encoder1.__dict__
    assert "forward" not in model.decoder2.__dict__
    assert enable_mer_only(model) is model


def test_mer_only_outputs_gradients_rng_and_norm_buffers_match():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with candidate_execution():
            report = memory.model_equivalence(torch.device("cpu"))
        assert report["status"] == "PASS" and report["normalization_buffers_unchanged"]
    finally:
        torch.set_num_threads(old_threads)


def test_candidate_context_restores_baseline_even_after_failure():
    old_memory, old_pilot = memory.enable_checkpointing, pilot.enable_checkpointing
    with pytest.raises(RuntimeError):
        with candidate_execution():
            assert memory.enable_checkpointing is enable_mer_only
            assert pilot.enable_checkpointing is enable_mer_only
            raise RuntimeError("intentional check failure")
    assert memory.enable_checkpointing is old_memory and pilot.enable_checkpointing is old_pilot
