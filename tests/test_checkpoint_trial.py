"""Execution-only checkpoint trial: numerical equivalence and saved activations."""
from copy import deepcopy
import types

import torch

from tools.merit_checkpoint_trial import checkpointed_mer, mer_equivalence, model_equivalence
from merit_covid.models.merit_unetrpp import MasterEvidenceRouting


def test_mer_checkpoint_outputs_and_all_gradients_match():
    assert mer_equivalence()["status"] == "PASS"


def test_full_model_checkpoint_outputs_gradients_and_dropout_match():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        assert model_equivalence(torch.device("cpu"))["status"] == "PASS"
    finally:
        torch.set_num_threads(old)


def test_mer_checkpoint_reduces_unique_saved_tensor_storage():
    torch.manual_seed(31)
    module = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3, query_chunk_size=8).train()
    trial = deepcopy(module)
    trial.forward = types.MethodType(checkpointed_mer, trial)
    def saved_bytes(model):
        sizes = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            sizes[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model(torch.randn(1, 4, 3, 8, 8, requires_grad=True), [1])
        total = sum(sizes.values())
        output.sum().backward()
        return total
    before, after = saved_bytes(module), saved_bytes(trial)
    assert after < before / 2, (before, after)
