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
