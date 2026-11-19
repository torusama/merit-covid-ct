"""BN statistics stay fixed through training, evaluation and checkpoint resume."""
from copy import deepcopy

import pytest
import torch
import yaml

from merit_covid.engine.normalization import enable_frozen_batchnorm_stats
from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import load_checkpoint_model, protocol_fingerprint, validate_experiment
from scripts.train_colab import train


def bn_buffers(model):
    return {name:value.clone() for name,value in model.state_dict().items()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def test_training_freezes_statistics_but_keeps_affine_gradients_and_dropout():
    torch.manual_seed(91)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                torch.nn.Dropout(.25), torch.nn.Linear(4, 1))
    # Populate meaningful running statistics before freezing them.
    model(torch.randn(8, 3))
    before = bn_buffers(model)
    identities = {name:id(value) for name,value in model.state_dict(keep_vars=True).items()}
    enable_frozen_batchnorm_stats(model)
    enable_frozen_batchnorm_stats(model)
    assert len(model._forward_pre_hooks) == 1
    assert identities == {name:id(value) for name,value in model.state_dict(keep_vars=True).items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial_affine = model[1].weight.detach().clone()
    for _ in range(2):
        model.eval(); model.train()  # Like evaluation followed by the next epoch.
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(8, 3)).square().mean().backward()
        assert model[2].training and not model[1].training
        assert model[1].weight.requires_grad and model[1].weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(initial_affine, model[1].weight)
    for name,value in before.items():
        assert torch.equal(value, model.state_dict()[name])


def test_stateless_batchnorm_cannot_claim_frozen_running_statistics():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3, track_running_stats=False))
    with pytest.raises(ValueError, match='saved running mean'):
        enable_frozen_batchnorm_stats(model)


@pytest.mark.parametrize('value', ['true', 1, None])
def test_freeze_setting_requires_a_boolean(value):
    config = dict(data={'manifest':'/tmp/a.json', 'split_path':'/tmp/b.json'},
        preprocess={'slab_size':11, 'size_hw':[32,32]}, model={'img_size':[11,32,32]},
        loss={}, training={'freeze_batchnorm_stats':value})
    with pytest.raises(ValueError, match='freeze_batchnorm_stats must be boolean'):
        validate_experiment(config)


def test_frozen_bn_is_protocol_bound_and_disabled_setting_keeps_legacy_hash(tmp_path):
    config = validate_experiment(synthetic_experiment(tmp_path/'data'))
    legacy = deepcopy(config); legacy['training'].pop('freeze_batchnorm_stats')
    assert protocol_fingerprint(config, {}) == protocol_fingerprint(legacy, {})
    frozen = deepcopy(config); frozen['training']['freeze_batchnorm_stats'] = True
    assert protocol_fingerprint(frozen, {}) != protocol_fingerprint(config, {})


def test_full_trainer_restores_policy_and_matches_mid_epoch_resume(tmp_path):
    config = synthetic_experiment(tmp_path/'data', device='cpu')
    config['training'].update(batch_size=1, activation_checkpointing=True,
                              factorized_mer=True, freeze_batchnorm_stats=True)
    path = tmp_path/'experiment.yaml'; path.write_text(yaml.safe_dump(config))
    full, _ = train(path, 0, tmp_path/'full', epochs=1)
    expected = {name:value.clone() for name,value in full.model.state_dict().items()}
    frozen = bn_buffers(full.model)
    assert frozen and all(value.item() == 0 for name,value in frozen.items() if name.endswith('num_batches_tracked'))
    del full
    _, history = train(path, 0, tmp_path/'resume', epochs=1, checkpoint_every=1, max_seconds=1e-8)
    assert not history
    resumed, history = train(path, 0, tmp_path/'resume', epochs=1,
                             resume=tmp_path/'resume/in_progress.pt')
    assert len(history) == 1
    assert all(torch.equal(value, resumed.model.state_dict()[name]) for name,value in expected.items())
    model, saved = load_checkpoint_model(tmp_path/'resume/last.pt')
    assert saved['config']['training']['freeze_batchnorm_stats'] is True
    assert model._frozen_batchnorm_stats_enabled
    model.train()
    with torch.no_grad():
        model(torch.randn(1, 1, 11, 32, 32), [5])
    assert all(torch.equal(value, model.state_dict()[name]) for name,value in frozen.items())
    invalid = deepcopy(config); invalid['training']['freeze_batchnorm_stats'] = False
    path.write_text(yaml.safe_dump(invalid))
    with pytest.raises(ValueError, match='mismatch'):
        train(path, 0, tmp_path/'wrong-policy', epochs=1, resume=tmp_path/'resume/last.pt')
