"""Run against the exact deployed frozen source, as the migration tests do."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import TRAIN_DEFAULTS, load_experiment, split_datasets, validate_experiment, protocol_fingerprint
from scripts.train_colab import train
from tools.epoch1_low_lr_pilot import branch_checkpoint, run

pytestmark = pytest.mark.skipif('mer_recomputation' not in TRAIN_DEFAULTS,
    reason='Execute inside the frozen pilot test_source')


def test_pilot_stops_at_batch_budget_and_preserves_parent(tmp_path):
    config = synthetic_experiment(tmp_path/'data', device='cpu')
    config['training'].update(activation_checkpointing=True, batch_size=1, shuffle_mode='volume')
    path = tmp_path/'source.yaml'; path.write_text(yaml.safe_dump(config))
    train(path, 0, tmp_path/'parent', epochs=1)
    parent = tmp_path/'parent/last.pt'
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    original = torch.load(parent, map_location='cpu', weights_only=False)
    datasets, _ = split_datasets(load_experiment(path), 0)
    hashes = {case.case_id:{key:hashlib.sha256(gzip.decompress(Path(source).read_bytes())).hexdigest()
        for key,source in [('image',case.image),('mask',case.mask)]}
        for dataset in datasets.values() for case in dataset.cases}
    output = tmp_path/'pilot'
    report = run(parent, tmp_path/'parent/history.jsonl', hashes, output,
                 total_batches=2, evaluate_every=1, learning_rate=3e-5)
    assert len(report['train']) == 2
    assert [row['batch'] for row in report['probes']] == [0, 1, 2]
    saved = torch.load(output/'checkpoints/in_progress.pt', map_location='cpu', weights_only=False)
    assert saved['epoch'] == 1 and saved['in_progress']['batches'] == 2
    assert saved['in_progress']['steps'] == 2 and saved['in_progress']['skipped'] == 0
    assert saved['protocol_fingerprint'] == protocol_fingerprint(saved['config'], saved['provenance'])
    assert all(group['lr'] == 3e-5 for group in saved['optimizer']['param_groups'])
    assert saved['scheduler']['_last_lr'] == [3e-5]
    assert saved['scheduler']['best'] == original['scheduler']['best']
    assert saved['execution_migration']['optimizer_moments_preserved']
    assert saved['execution_migration']['learning_rate_changed']
    assert saved['execution_migration']['optimizer_reset'] is False
    for key,state in original['optimizer']['state'].items():
        assert saved['optimizer']['state'][key]['step'].item() == state['step'].item()+2
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_hash
    assert not (output/'checkpoints/last.pt').exists()
    status = json.loads((output/'run-status.json').read_text())
    assert status['status'] == 'pilot_complete' and status['full_epoch_complete'] is False


@pytest.mark.parametrize('change', ['increase_lr', 'loss', 'sampling'])
def test_branch_rejects_unrelated_changes(change):
    old = validate_experiment(dict(data={'manifest':'/tmp/a.json', 'split_path':'/tmp/b.json'},
        preprocess={'slab_size':11, 'size_hw':[32,32]}, loss={}, model={'img_size':[11,32,32]},
        training={'epochs':1, 'activation_checkpointing':True}))
    base = deepcopy(old)
    base['training'].update(epochs=2, deterministic=False, factorized_mer=True, mer_recomputation=False)
    base['model']['query_chunk_size'] = 4096
    target = deepcopy(base); target['training']['learning_rate'] = 3e-5
    if change == 'increase_lr': target['training']['learning_rate'] = 3e-3
    if change == 'loss': target['loss']['presence_weight'] = .9
    if change == 'sampling': target['training']['shuffle_mode'] = 'volume'
    with pytest.raises(ValueError, match='only lower learning rate'):
        branch_checkpoint({}, base, target, {})
