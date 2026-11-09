"""Train one saved MERIT fold with cache-friendly case-grouped shuffling.

One worker and a shared seeded generator make each volume load once per epoch
and preserve epoch-boundary Trainer resume. No test evaluation or data download.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Sampler

from merit_covid.data.pipeline.dataloader import collate_slabs
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


def train(config_path, fold, output, *, epochs=None, resume=None):
    config = load_experiment(config_path)
    if epochs is not None:
        config["training"]["epochs"] = epochs
        config = validate_experiment(config)
    settings = config["training"]
    if settings["num_workers"] != 0:
        raise ValueError("Case-grouped one-volume caching requires num_workers=0")
    datasets, provenance = split_datasets(config, fold)
    provenance["train_order"] = "seeded case shuffle, then seeded master shuffle within case; every master once"
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(datasets["train"], batch_size=settings["batch_size"],
        sampler=CaseGroupedSampler(datasets["train"], generator), generator=generator,
        num_workers=0, collate_fn=collate_slabs)
    trainer = Trainer(config, loader, datasets["validation"].cases, provenance, output)
    return trainer, trainer.fit(resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/merit_colab_256.yaml")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Total epoch budget; can be extended on resume")
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint from this training entry point")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.config, args.fold, args.output, epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()
