"""Opt-in production execution, protocol binding and exact mid-epoch resume."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import load_checkpoint_model, protocol_fingerprint, validate_experiment
from merit_covid.engine.factorized_attention import factorized_mer
from scripts.train_colab import train


def test_production_candidate_matches_t4_tested_implementation():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/merit_covid/engine/factorized_attention.py").read_bytes() == (
        root / "tools/merit_factorized_attention.py").read_bytes()


def test_kaggle_guard_validates_production_mer_on_cpu():
    from scripts.kaggle.factorized_guard import check
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        results = check("cpu")
    finally:
        torch.set_num_threads(old)
    assert len(results) == 2 and all(item["status"] == "PASS" for item in results)


def test_opt_in_is_protocol_bound_and_legacy_fingerprint_unchanged(tmp_path):
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    assert config["training"]["factorized_mer"] is False
    legacy = deepcopy(config)
    legacy["training"].pop("factorized_mer")
    assert protocol_fingerprint(config, {}) == protocol_fingerprint(legacy, {})
    candidate = deepcopy(config)
    candidate["training"].update(factorized_mer=True, activation_checkpointing=True)
    candidate = validate_experiment(candidate)
    assert protocol_fingerprint(candidate, {}) != protocol_fingerprint(config, {})
    for override in ({"activation_checkpointing": False}, {"factorized_mer": "yes"}):
        invalid = deepcopy(candidate)
        invalid["training"].update(override)
        with pytest.raises(ValueError):
            validate_experiment(invalid)
    invalid = deepcopy(candidate)
    invalid["model"]["use_mer"] = False
    with pytest.raises(ValueError, match="requires"):
        validate_experiment(invalid)


def test_factorized_trainer_mid_epoch_resume_and_inference(tmp_path):
    config = synthetic_experiment(tmp_path / "data", device="cpu")
    config["training"].update(batch_size=1, augmentation_flips=True,
                              activation_checkpointing=True, factorized_mer=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    uninterrupted, _ = train(path, 0, tmp_path / "whole", epochs=1)
    expected = {k: v.detach().clone() for k, v in uninterrupted.model.state_dict().items()}
    del uninterrupted
    _, history = train(path, 0, tmp_path / "resume", epochs=1,
                       checkpoint_every=1, max_seconds=0.000001)
    checkpoint = tmp_path / "resume/in_progress.pt"
    assert not history and checkpoint.is_file()
    resumed, history = train(path, 0, tmp_path / "resume", epochs=1,
                             checkpoint_every=2, resume=checkpoint)
    assert len(history) == 1 and not checkpoint.exists()
    assert all(torch.equal(expected[k], v) for k, v in resumed.model.state_dict().items())
    model, saved = load_checkpoint_model(tmp_path / "resume/last.pt")
    assert saved["config"]["training"]["factorized_mer"] is True
    assert model.mer.forward.__func__ is factorized_mer
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
