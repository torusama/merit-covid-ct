"""One real CT crop -> UNETR++ forward, no training or performance evaluation.

Default reads A0025 in fold 0/train, forwards ONE patch and reports software
checks. It never treats an incomplete set of patches as a volume prediction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from merit_covid.data.pipeline.preprocessing import read_arrays
from merit_covid.data.unetrpp_input import PatchConfig, UNETRPPPatchDataset, make_patch_dataloader
from merit_covid.engine.patch_inference import predict_patch_batch
from merit_covid.models.unetrpp import UNETRPP

ROOT = Path(__file__).resolve().parents[1]


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/covid19_20/cases.json")
    parser.add_argument("--split", type=Path, default=ROOT / "splits/covid19_20_5fold.json")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--case-id", default="volume-covid19-A-0025")
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--overlap", type=float, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--assume-mm", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON software-check report; no tensors/checkpoints saved")
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(42)
    config = PatchConfig(overlap=tuple(args.overlap) if args.overlap else None, assume_mm=args.assume_mm)
    dataset = UNETRPPPatchDataset.from_split(
        args.manifest, args.split, args.fold, args.partition, config=config,
        require_mask=False, case_ids=[args.case_id],
    )
    if not 0 <= args.patch_index < len(dataset):
        parser.error(f"--patch-index must be in 0..{len(dataset)-1}")
    case = dataset.cases[0]  # Inference Dataset already removed the GT path.
    stat_before = (case.image.stat().st_size, case.image.stat().st_mtime_ns)
    from torch.utils.data import Subset
    batch = next(iter(make_patch_dataloader(Subset(dataset, [args.patch_index]), num_workers=args.workers)))
    location = dataset.plans[0][args.patch_index]
    # Independently call the shared reader to verify the real crop values/order.
    native, mask, _ = read_arrays(case, dataset.preprocess, require_mask=False)
    np.testing.assert_array_equal(batch["image_patch"][0, 0].numpy()[location.valid_slices],
                                  native[location.source_slices])
    assert mask is None
    del native
    model = UNETRPP(in_channels=1, out_channels=1, do_ds=True).eval()
    before = parameter_digest(model)
    probability = predict_patch_batch(model, batch)
    weights_unchanged = before == parameter_digest(model)
    if not weights_unchanged or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected parameter/gradient change")
    stat_after = (case.image.stat().st_size, case.image.stat().st_mtime_ns)
    if stat_before != stat_after:
        raise RuntimeError("Source CT size/timestamp changed")
    report = {
        "purpose": "real CT -> native crop -> untrained UNETR++ forward software check",
        "case_id": case.case_id, "fold": args.fold, "partition": args.partition,
        "volume_shape_zyx": dataset.plans[0].shape_zyx,
        "patches_to_cover_case": len(dataset), "forwarded_patches": 1,
        "patch": location.to_dict(), "input_shape": list(batch["image_patch"].shape),
        "input_dtype": str(batch["image_patch"].dtype),
        "input_range": [float(batch["image_patch"].min()), float(batch["image_patch"].max())],
        "real_voxels_in_patch": int(batch["valid_region"].sum()),
        "crop_equals_shared_native_reader": True, "ground_truth_read": False,
        "probability_shape": list(probability.shape), "probability_dtype": str(probability.dtype),
        "probability_finite": bool(np.isfinite(probability).all()),
        "probability_range": [float(probability.min()), float(probability.max())],
        "weights_unchanged": weights_unchanged, "parameter_sha256": before,
        "gradients_created": False, "source_size_and_timestamp_unchanged": True,
        "input_policy": config.to_dict(), "selection": dataset.selection,
        "full_case_prediction_created": False, "training_or_metrics_performed": False,
        "note": "Random initialization; probability range has no model-quality meaning. One patch is not a whole CT prediction.",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
