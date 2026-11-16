"""Case-grouped train order and epoch-boundary replay."""

import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from scripts.train_colab import CaseGroupedSampler, train


def test_colab_config_records_spatial_assumption_without_hu_window():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/merit_colab_256.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["preprocess"]["assume_mm"] is True
    assert config["preprocess"]["intensity_encoding"] == "native"
    assert "hu_window" not in config["preprocess"]
    assert config["preprocess"]["size_hw"] == [256, 256]
    assert config["model"]["img_size"] == [11, 256, 256]


def test_case_grouped_sampler_visits_every_master_and_replays_generator():
    class DatasetIndex:
        cases = ["a", "b", "c"]
        ends = [0, 2, 5, 9]
        def __len__(self):
            return self.ends[-1]
    dataset = DatasetIndex()
    generator = torch.Generator().manual_seed(42)
    sampler = CaseGroupedSampler(dataset, generator)
    first = list(sampler)
    assert sorted(first) == list(range(9))
    cases = [0 if x < 2 else 1 if x < 5 else 2 for x in first]
    assert len([i for i in range(len(cases)) if i == 0 or cases[i] != cases[i - 1]]) == 3
    saved = generator.get_state()
    following = list(sampler)
    replay_generator = torch.Generator()
    replay_generator.set_state(saved)
    assert list(CaseGroupedSampler(dataset, replay_generator)) == following


def test_grouped_training_resume_matches_uninterrupted(tmp_path):
    config = synthetic_experiment(tmp_path / "data", device="cpu")
    config["training"]["batch_size"] = 1
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    uninterrupted, history = train(path, 0, tmp_path / "full", epochs=2)
    assert len(history) == 2
    expected = {k: v.detach().clone() for k, v in uninterrupted.model.state_dict().items()}
    del uninterrupted

    partial, history = train(path, 0, tmp_path / "resumed", epochs=1)
    assert len(history) == 1
    del partial
    resumed, history = train(path, 0, tmp_path / "resumed", epochs=2,
                             resume=tmp_path / "resumed/last.pt")
    assert len(history) == 1 and resumed.start_epoch == 1
    assert all(torch.equal(expected[k], v) for k, v in resumed.model.state_dict().items())


def test_mid_epoch_checkpoint_replays_sampler_and_matches_full_run(tmp_path):
    config = synthetic_experiment(tmp_path / "data", device="cpu")
    config["training"].update(batch_size=1, augmentation_flips=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    whole, _ = train(path, 0, tmp_path / "whole", epochs=1)
    expected = {k: v.detach().clone() for k, v in whole.model.state_dict().items()}
    del whole
    _, history = train(path, 0, tmp_path / "resume", epochs=1,
                       checkpoint_every=1, max_seconds=0.000001)
    checkpoint = tmp_path / "resume" / "in_progress.pt"
    assert not history and checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["in_progress"]["batches"] == 1
    # Kaggle remounts can change timestamps without changing source bytes.
    import gzip, hashlib, json, os
    records = json.loads((tmp_path / "data/cases.json").read_text())
    hashes = {r['id']: {key:hashlib.sha256(gzip.open(tmp_path/'data'/r[key],'rb').read()).hexdigest()
                         for key in ('image','mask')} for r in records}
    hashes_path = tmp_path/'hashes.json'
    hashes_path.write_text(json.dumps(hashes))
    source = tmp_path/'data'/records[0]['image']
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000000))
    resumed, history = train(path, 0, tmp_path / "resume", epochs=1, resume=checkpoint,
                             checkpoint_every=2, verified_resume_hashes=hashes_path)
    assert len(history) == 1 and not checkpoint.exists()
    assert all(torch.equal(expected[k], v) for k, v in resumed.model.state_dict().items())


def test_sampler_replay_does_not_reload_completed_samples():
    from torch.utils.data import DataLoader, Dataset
    from merit_covid.data.pipeline.dataloader import resume_loader_iterator
    class RecordedDataset(Dataset):
        def __init__(self): self.reads = []
        def __len__(self): return 17
        def __getitem__(self, index):
            self.reads.append(index)
            return index
    ds = RecordedDataset()
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(ds, batch_size=2, shuffle=True, generator=generator)
    epoch_state = generator.get_state().clone()
    iterator = iter(loader)
    for _ in range(3): next(iterator)
    checkpoint_state = generator.get_state().clone()
    expected = torch.cat(list(iterator))
    final_state = generator.get_state().clone()
    ds.reads.clear()
    generator.set_state(epoch_state)
    actual = torch.cat(list(resume_loader_iterator(loader,3,checkpoint_state,lambda:None)))
    assert torch.equal(actual,expected)
    assert ds.reads == expected.tolist()
    assert torch.equal(generator.get_state(),final_state)
