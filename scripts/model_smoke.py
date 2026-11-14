"""Model-only synthetic forward/backward; no optimizer, epochs or checkpoints."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unetrpp", "merit"), required=True)
    parser.add_argument("--config", type=Path, help="Model-only YAML; omit for compact CPU geometry")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = {}
    if args.config:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if (not isinstance(config, dict) or not isinstance(config.get("model"), dict)
                or set(config) - {"model", "loss"}):
            parser.error("Expected model-only YAML with optional loss mapping")
    if args.model == "merit":
        from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
        kwargs = config.get("model", {"img_size": (11, 32, 32), "feature_size": 2, "depths": (1, 1, 1, 1)})
        model = MERITUNETRPP(**kwargs)
    else:
        from merit_covid.models.unetrpp import UNETRPP
        kwargs = config.get("model", {"img_size": (32, 32, 32), "feature_size": 2, "do_ds": True})
        model = UNETRPP(**kwargs)
    model.train(args.backward)
    image = torch.randn(1, model.in_channels, *model.img_size)
    with torch.set_grad_enabled(args.backward):
        if args.model == "merit":
            masters = torch.tensor([model.img_size[0] // 2])
            output = model(image, masters)
            logits = [output.logits]
            loss = MERITLoss(**config.get("loss", {}))(output, torch.zeros_like(image), masters)["total"] if args.backward else None
        else:
            output = model(image)
            logits = output if isinstance(output, list) else [output]
            loss = sum(t.square().mean() for t in logits) if args.backward else None
        if not all(torch.isfinite(t).all() for t in logits):
            raise RuntimeError("Nonfinite output")
        if args.backward:
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing or nonfinite gradients")
    print(json.dumps({"synthetic_only": True, "model": args.model, "input": list(image.shape),
                      "outputs": [list(t.shape) for t in logits], "backward_checked": args.backward,
                      "weights_updated": False, "parameters": sum(p.numel() for p in model.parameters())}, indent=2))


if __name__ == "__main__":
    main()
