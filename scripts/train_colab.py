"""Train one saved MERIT fold with cache-friendly case-grouped shuffling.

One worker and a shared seeded generator make each volume load once per epoch
and preserve epoch-boundary Trainer resume. No test evaluation or data download.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Sampler

from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.engine.training import Trainer, load_experiment, split_datasets, validate_experiment


class CaseGroupedSampler(Sampler):
    """Shuffle cases then masters within each case, visiting every master once."""

    def __init__(self, dataset, generator):
        self.dataset = dataset
        self.generator = generator
        if len(dataset.ends) != len(dataset.cases) + 1 or dataset.ends[0] != 0:
            raise ValueError("Dataset must provide the shared cumulative case index")
        if any(a >= b for a, b in zip(dataset.ends, dataset.ends[1:])):
            raise ValueError("Every case must contain at least one master slice")

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for case_index in torch.randperm(len(self.dataset.cases), generator=self.generator).tolist():
            start, end = self.dataset.ends[case_index:case_index + 2]
            for offset in torch.randperm(end - start, generator=self.generator).tolist():
                yield start + offset


def train(config_path, fold, output, *, epochs=None, resume=None, checkpoint_every=0, max_seconds=None,
          verified_resume_hashes=None):
    config = load_experiment(config_path)
    if verified_resume_hashes is not None and config["data"].get("mode") == "tiling128_v3_fold0":
        raise ValueError("Legacy raw-volume timestamp rebinding cannot verify all sixteen tiles; use a matching Tiling128 checkpoint without --verified-resume-hashes")
    config["training"]["shuffle_mode"] = "volume"
    if epochs is not None:
        config["training"]["epochs"] = epochs
    config = validate_experiment(config)
    settings = config["training"]
    if settings["num_workers"] != 0:
        raise ValueError("Case-grouped one-volume caching requires num_workers=0")
    datasets, provenance = split_datasets(config, fold)
    provenance["train_order"] = ("volume-balanced sampling with replacement" if settings["samples_per_volume"] else
                                 "seeded case shuffle, then seeded master shuffle within case; every master once")
    loader = make_dataloader(datasets["train"], batch_size=settings["batch_size"],
        shuffle=True, shuffle_mode="volume", seed=settings["seed"], num_workers=0,
        samples_per_volume=settings["samples_per_volume"])
    trainer = Trainer(config, loader, datasets["validation"].cases, provenance, output)
    if verified_resume_hashes is not None:
        if resume is None:
            raise ValueError("Verified resume hashes require a checkpoint")
        import json
        from scripts.kaggle.resume_checkpoint import verify_dataset_content, rebind_verified_timestamps
        expected = json.loads(Path(verified_resume_hashes).read_text(encoding="utf-8"))
        all_cases = sorted([c for dataset in datasets.values() for c in dataset.cases], key=lambda c:c.case_id)
        verify_dataset_content(all_cases, expected)
        saved = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)
        resume = rebind_verified_timestamps(saved, trainer.config, trainer.provenance, trainer.fingerprint)
        print("Checkpoint source identity verified; continuing saved batch position", flush=True)
    return trainer, trainer.fit(resume=resume, checkpoint_every=checkpoint_every, max_seconds=max_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/merit_colab_256.yaml")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Total epoch budget; can be extended on resume")
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint from this training entry point")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save in-progress state every N batches; 0 disables")
    parser.add_argument("--max-seconds", type=float, help="Stop safely at a batch boundary after this training time")
    parser.add_argument("--verified-resume-hashes", type=Path, help="Full content hashes required to rebind Kaggle source timestamps")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.config, args.fold, args.output, epochs=args.epochs, resume=args.resume,
          checkpoint_every=args.checkpoint_every, max_seconds=args.max_seconds,
          verified_resume_hashes=args.verified_resume_hashes)


if __name__ == "__main__":
    main()
