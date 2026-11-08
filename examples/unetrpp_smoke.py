"""Synthetic UNETR++ forward/backward only; no data, optimizer or checkpoints."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from merit_covid.models.unetrpp import UNETRPP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Model YAML; omit for a small CPU-only test")
    parser.add_argument("--backward", action="store_true", help="Check gradients without updating weights")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    if args.config:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) != {"model"} or not isinstance(config["model"], dict):
            parser.error("Config must contain exactly one 'model' parameter mapping")
        kwargs = config["model"]
    else:
        # Test configuration, NOT the published Synapse experiment configuration.
        kwargs = {"img_size": (32, 32, 32), "feature_size": 2, "do_ds": True}
    model = UNETRPP(**kwargs)
    model.train(args.backward)
    x = torch.randn(1, model.in_channels, *model.img_size)
    with torch.set_grad_enabled(args.backward):
        output = model(x)
        logits = output if isinstance(output, list) else [output]
        if not all(torch.isfinite(t).all() for t in logits):
            raise RuntimeError("Nonfinite output")
        if args.backward:
            sum(t.square().mean() for t in logits).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing or nonfinite gradients")
    print(json.dumps({
        "synthetic_only": True,
        "configuration": "explicit YAML" if args.config else "small CPU test (not experiment settings)",
        "input": list(x.shape), "outputs": [list(t.shape) for t in logits],
        "parameters": sum(p.numel() for p in model.parameters()),
        "backward_checked": args.backward, "weights_updated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
