"""Reproduce the runtime/data-binding failures found during the MERIT review.

Only synthetic NIfTI and compact models are used. These checks do not establish
paper equivalence, clinical quality, or the correctness of the resized input.
"""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import pytest
import torch

from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.engine.slab_inference import predict_slab_case
from merit_covid.engine.synthetic import synthetic_experiment
from merit_covid.engine.training import (
    Trainer, load_checkpoint_model, protocol_fingerprint, source_identity,
    split_datasets, validate_experiment,
)
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITUNETRPPAdapter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    config = validate_experiment(synthetic_experiment(tmp_path / "data"))
    yield config
    torch.set_num_threads(old)


def make_trainer(config, output):
    datasets, provenance = split_datasets(config, 0)
    settings = config["training"]
    loader = make_dataloader(datasets["train"], batch_size=2, shuffle=True,
                             shuffle_mode=settings["shuffle_mode"], seed=settings["seed"])
    return Trainer(config, loader, datasets["validation"].cases, provenance, output)


@pytest.mark.parametrize("threshold", [0., 1., float("nan"), float("inf"), -.1, 1.1])
def test_threshold_fails_before_training(experiment, threshold):
    experiment["training"]["threshold"] = threshold
    with pytest.raises(ValueError, match="[Tt]hreshold"):
        validate_experiment(experiment)


def test_resume_rejects_redirected_gt_with_same_basenames(experiment, tmp_path):
    trainer = make_trainer(experiment, tmp_path / "original")
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint, 0)
    manifest = Path(experiment["data"]["manifest"])
    records = json.loads(manifest.read_text())
    alternate = manifest.parent / "alternate_gt"
    alternate.mkdir()
    for record in records:
        old_path = manifest.parent / record["mask"]
        image = nib.load(old_path)
        nib.save(nib.Nifti1Image(1 - image.get_fdata(), image.affine, image.header.copy()),
                 alternate / old_path.name)
        record["mask"] = f"alternate_gt/{old_path.name}"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    fresh = make_trainer(experiment, tmp_path / "redirected")
    # Split membership/basename identity is unchanged; source binding is not.
    assert fresh.provenance["split_fingerprint"] == trainer.provenance["split_fingerprint"]
    assert fresh.fingerprint != trainer.fingerprint
    with pytest.raises(ValueError, match="mismatch"):
        fresh.resume(checkpoint)


def test_in_place_gt_edit_and_manifest_edit_invalidate_binding(experiment, tmp_path):
    datasets, provenance = split_datasets(experiment, 0)
    mask = datasets["train"].cases[0].mask
    stat = mask.stat()
    image = nib.load(mask)
    nib.save(nib.Nifti1Image(1 - image.get_fdata(), image.affine, image.header.copy()), mask)
    # Guarantee a changed timestamp even on coarse filesystem clocks.
    os.utime(mask, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    loader = make_dataloader(datasets["train"], shuffle=True)
    with pytest.raises(ValueError, match="changed after split indexing"):
        Trainer(experiment, loader, datasets["validation"].cases, provenance, tmp_path / "stale")
    current = source_identity(experiment, datasets["train"].cases)
    manifest = Path(experiment["data"]["manifest"])
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    assert source_identity(experiment, datasets["train"].cases) != current


def test_trainer_rejects_gt_outside_current_manifest(experiment, tmp_path):
    datasets, provenance = split_datasets(experiment, 0)
    case = datasets["train"].cases[0]
    alternate = tmp_path / case.mask.name
    shutil.copy2(case.mask, alternate)
    datasets["train"].cases[0] = replace(case, mask=alternate)
    loader = make_dataloader(datasets["train"], shuffle=True)
    with pytest.raises(ValueError, match="differs from the current manifest"):
        Trainer(experiment, loader, datasets["validation"].cases, provenance, tmp_path / "bad")


def test_protocol_keeps_data_config_and_only_extends_epochs(experiment, tmp_path):
    trainer = make_trainer(experiment, tmp_path / "run")
    changed = deepcopy(experiment)
    changed["data"]["manifest"] += ".different"
    assert protocol_fingerprint(changed, trainer.provenance) != trainer.fingerprint
    checkpoint = tmp_path / "longer.pt"
    trainer.config["training"]["epochs"] = 3
    trainer.save_checkpoint(checkpoint, 0)
    with pytest.raises(ValueError, match="cannot reduce"):
        make_trainer(experiment, tmp_path / "shorter").resume(checkpoint)


def test_legacy_checkpoint_readable_for_inference_but_not_resume(experiment, tmp_path):
    trainer = make_trainer(experiment, tmp_path / "run")
    checkpoint = tmp_path / "legacy.pt"
    trainer.save_checkpoint(checkpoint, 0)
    payload = torch.load(checkpoint, weights_only=False)
    payload["schema_version"] = 1
    payload["config"]["training"].pop("shuffle_mode")
    payload["provenance"].pop("source_identity")
    payload["provenance"].pop("all_mask_paths")
    payload["protocol_fingerprint"] = protocol_fingerprint(payload["config"], payload["provenance"], version=1)
    torch.save(payload, checkpoint)
    loaded, _ = load_checkpoint_model(checkpoint)
    assert not loaded.training
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="unsafe legacy resume"):
        trainer.resume(checkpoint)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_master_logits_rejected_before_sigmoid(experiment, monkeypatch, value):
    datasets, _ = split_datasets(experiment, 0)
    case = datasets["test"].cases[0]
    model = MERITUNETRPP(**experiment["model"]).eval()
    def invalid(self, batch):
        image = batch["image_slab"]
        return torch.full((len(image), 1, *image.shape[-2:]), value)
    monkeypatch.setattr(MERITUNETRPPAdapter, "predict_logits", invalid)
    with pytest.raises(ValueError, match="before sigmoid"):
        predict_slab_case(model, case, datasets["test"].config)


@pytest.mark.parametrize("workers", [0, 1])
def test_volume_shuffle_visits_all_masters_once_and_replays_next_epoch(experiment, workers):
    datasets, _ = split_datasets(experiment, 0)
    dataset = datasets["train"]
    def loader(seed):
        return make_dataloader(dataset, shuffle=True, shuffle_mode="volume", seed=seed,
                               batch_size=1, num_workers=workers)
    def order(source):
        return [(b["case_id"][0], int(b["master_slice_index"][0])) for b in source]
    first = loader(13)
    seen = order(first)
    expected = {(c.case_id, z) for c, (_, _, meta) in
                zip(dataset.cases, [dataset._load_case(i) for i in range(len(dataset.cases))])
                for z in range(meta["canonical_shape_xyz"][2])}
    assert len(seen) == len(set(seen)) == len(dataset) and set(seen) == expected
    assert sum(i == 0 or seen[i][0] != seen[i-1][0] for i in range(len(seen))) == len(dataset.cases)
    assert seen == order(loader(13))
    state = first.generator.get_state()
    next_epoch = order(first)
    restored = loader(999)
    restored.generator.set_state(state)
    assert next_epoch == order(restored)
    assert next_epoch != seen


def test_loader_mode_frozen_and_eval_shuffle_forbidden(experiment, tmp_path):
    datasets, provenance = split_datasets(experiment, 0)
    loader = make_dataloader(datasets["train"], shuffle=True, shuffle_mode="volume")
    with pytest.raises(ValueError, match="frozen shuffle_mode"):
        Trainer(experiment, loader, datasets["validation"].cases, provenance, tmp_path / "bad")
    for partition in ("validation", "test"):
        with pytest.raises(ValueError, match="deterministic slice order"):
            make_dataloader(datasets[partition], shuffle=True, shuffle_mode="volume")


def test_volume_shuffle_fills_single_volume_cache_once_per_case(experiment, monkeypatch):
    from merit_covid.data.pipeline import dataset as module
    datasets, _ = split_datasets(experiment, 0)
    original = module.read_volume
    loaded = []
    def counted(case, *args, **kwargs):
        loaded.append(case.case_id)
        return original(case, *args, **kwargs)
    monkeypatch.setattr(module, "read_volume", counted)
    dataset = datasets["train"]
    for _ in make_dataloader(dataset, shuffle=True, shuffle_mode="volume", num_workers=0):
        pass
    assert len(loaded) == len(set(loaded)) == len(dataset.cases)


@pytest.mark.parametrize("model", ["merit", "unetrpp"])
def test_documented_model_smoke_forward_backward(model):
    result = subprocess.run([sys.executable, "scripts/model_smoke.py", "--model", model, "--backward"],
                            cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(result.stdout)
    assert report["synthetic_only"] and report["backward_checked"] and not report["weights_updated"]


def test_real_input_script_help_needs_no_dataset():
    result = subprocess.run([sys.executable, "scripts/unetrpp_real_input.py", "--help"],
                            cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
    assert "--case-id" in result.stdout and "--assume-mm" in result.stdout
