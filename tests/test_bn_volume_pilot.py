"""Controls must isolate BN updates and replay the real loader order."""
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import TRAIN_DEFAULTS, load_experiment, split_datasets
from merit_covid.data.pipeline.dataloader import make_dataloader
from scripts.train_colab import train
from tools.audit_bn_volume_pilot import case_pools, execute_arm, independent_resume, run

pytestmark = pytest.mark.skipif('mer_recomputation' not in TRAIN_DEFAULTS,
    reason='Execute against the frozen deployment test_source')


def test_replayed_case_pools_match_native_loader():
    class IndexedDataset:
        partition = 'train'
        ends = [0, 3, 7, 12]
        def __len__(self): return self.ends[-1]
        def __getitem__(self, index): return {'index': index, 'metadata': {}}
    dataset = IndexedDataset()
    loader = make_dataloader(dataset, shuffle=True, shuffle_mode='volume', seed=137)
    state = loader.generator.get_state().clone()
    pools = case_pools(dataset, state, 2)
    actual = [int(batch['index'][0]) for batch in loader]
    for pool in pools:
        matching = [index for index in actual if any(a <= pool[0] < b and a <= index < b
                    for a,b in zip(dataset.ends, dataset.ends[1:]))]
        assert pool == matching[:2]
    assert len(set(pools[0]+pools[1])) == 4


def test_each_arm_gets_independent_optimizer_tensors():
    source = {'optimizer': {'state': {0:{'step':torch.tensor(4.), 'exp_avg':torch.tensor([2.])}}},
              'scheduler': {'_last_lr':[3e-5]}}
    class FakeTrainer:
        def resume(self, cp):
            cp['optimizer']['state'][0]['step'].add_(1)
            cp['optimizer']['state'][0]['exp_avg'].add_(8)
    trainer = FakeTrainer()
    independent_resume(trainer, source)
    independent_resume(trainer, source)
    assert source['optimizer']['state'][0]['step'].item() == 4
    assert source['optimizer']['state'][0]['exp_avg'].item() == 2


def test_short_audit_isolates_bn_parameters_and_order(tmp_path):
    config = synthetic_experiment(tmp_path/'data', device='cpu')
    config['training'].update(activation_checkpointing=True, batch_size=1, shuffle_mode='volume')
    path = tmp_path/'source.yaml'; path.write_text(yaml.safe_dump(config))
    train(path, 0, tmp_path/'parent', epochs=1)
    parent = tmp_path/'parent/last.pt'
    original_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    datasets, _ = split_datasets(load_experiment(path), 0)
    hashes = {case.case_id:{key:hashlib.sha256(gzip.decompress(Path(source).read_bytes())).hexdigest()
        for key,source in [('image',case.image),('mask',case.mask)]}
        for dataset in datasets.values() for case in dataset.cases}
    result = run(parent, tmp_path/'parent/history.jsonl', hashes, tmp_path/'audit', per_case=1)
    arms = {arm['name']:arm for arm in result['arms']}
    assert len(arms) == 5 and all(arm['status'] == 'complete' for arm in arms.values())
    for name in ('bn_only_blocked', 'bn_only_two_cases_blocked', 'bn_only_two_cases_interleaved'):
        assert arms[name]['parameters_unchanged']
        assert arms[name]['optimizer_steps'] == 0
        assert arms[name]['bn_drift']['changed_buffers'] > 0
    frozen = arms['frozen_bn_train_blocked']
    assert frozen['optimizer_steps'] == 1 and frozen['parameters_unchanged'] is False
    assert frozen['bn_drift']['changed_buffers'] == 0
    assert arms['reference_train_blocked']['bn_drift']['batch_counter_increments'] == [1]
    blocked = arms['bn_only_two_cases_blocked']['batches']
    mixed = arms['bn_only_two_cases_interleaved']['batches']
    assert sorted((r['case'], r['master']) for r in blocked) == sorted((r['case'], r['master']) for r in mixed)
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == original_hash
