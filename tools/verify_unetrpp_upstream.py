"""Compare synthetic forward with pinned upstream architecture; no training.

The source directory must contain the official files recorded in upstream.json.
It is a local, ignored reference checkout, not an installed training framework.
Only import wiring/superclass/initializer are adapted in the reference namespace.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from merit_covid.models.unetrpp import UNETRPP


def load_reference(root):
    provenance = Path(__file__).resolve().parents[1] / "src/merit_covid/models/unetrpp/upstream.json"
    manifest = json.loads(provenance.read_text(encoding="utf-8"))
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned upstream source hash mismatch: {name}")
    base = root / "unetr_pp/network_architecture"

    def execute(path, replacements=(), inject=None):
        source = path.read_text(encoding="utf-8")
        for old, new in replacements:
            if old not in source:
                raise ValueError(f"Missing reference import: {old}")
            source = source.replace(old, new)
        namespace = {"__name__": f"reference_{path.stem}", **(inject or {})}
        exec(compile(source, str(path), "exec"), namespace)
        return namespace

    blocks = execute(base / "dynunet_block.py")
    layers = execute(base / "layers.py")
    transformer = execute(base / "synapse/transformerblock.py", [
        ("from unetr_pp.network_architecture.dynunet_block import UnetResBlock", ""),
    ], {"UnetResBlock": blocks["UnetResBlock"]})
    components = execute(base / "synapse/model_components.py", [
        ("from timm.models.layers import trunc_normal_", "from torch.nn.init import trunc_normal_"),
        ("from unetr_pp.network_architecture.layers import LayerNorm", ""),
        ("from unetr_pp.network_architecture.synapse.transformerblock import TransformerBlock", ""),
        ("from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock", ""),
    ], {"LayerNorm": layers["LayerNorm"], "TransformerBlock": transformer["TransformerBlock"],
        "get_conv_layer": blocks["get_conv_layer"], "UnetResBlock": blocks["UnetResBlock"]})
    model = execute(base / "synapse/unetr_pp_synapse.py", [
        ("from unetr_pp.network_architecture.neural_network import SegmentationNetwork",
         "from torch.nn import Module as SegmentationNetwork"),
        ("from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock", ""),
        ("from unetr_pp.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock", ""),
    ], {"UnetOutBlock": blocks["UnetOutBlock"], "UnetResBlock": blocks["UnetResBlock"],
        "UnetrPPEncoder": components["UnetrPPEncoder"], "UnetrUpBlock": components["UnetrUpBlock"]})
    return model["UNETR_PP"], manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional software verification JSON")
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference_class, manifest = load_reference(args.source_dir)
    kwargs = dict(in_channels=1, out_channels=1, img_size=(64, 128, 128),
                  feature_size=16, hidden_size=256, num_heads=4,
                  depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
    torch.manual_seed(42)
    reference = reference_class(**kwargs).eval()
    torch.manual_seed(42)
    model = UNETRPP(**kwargs).eval()
    original = reference.state_dict()
    actual = model.state_dict()
    if original.keys() != actual.keys():
        raise AssertionError("State-dict key mismatch")
    initialization_equal = all(torch.equal(original[k], actual[k]) for k in original)
    model.load_state_dict(original, strict=True)
    image = torch.randn(1, 1, 64, 128, 128)
    with torch.inference_mode():
        expected = reference(image)
        output = model(image)
    for a, b in zip(output, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    result = {
        "upstream_commit": manifest["commit"], "variant": "synapse",
        "synthetic_only": True, "input_shape": list(image.shape),
        "output_shapes": [list(t.shape) for t in output],
        "state_dict_keys_and_shapes_match": True,
        "same_seed_initialization_equal": initialization_equal,
        "strict_state_dict_loading": True,
        "all_three_outputs_bitwise_equal": True,
        "max_absolute_errors": [float((a - b).abs().max()) for a, b in zip(output, expected)],
        "parameters": sum(p.numel() for p in model.parameters()),
        "reference_adaptations": ["local import wiring", "SegmentationNetwork -> nn.Module",
                                  "timm trunc_normal_ -> torch.nn.init.trunc_normal_"],
        "torch": torch.__version__, "training_or_weight_update": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
