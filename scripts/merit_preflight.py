"""Train-only data audit and bounded real-CT MERIT pilot/GPU profile.

Run heavy commands sequentially. Reports are diagnostics, not clinical results.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("inspect")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--fold", type=int, default=0)
    audit.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    audit.add_argument("--assume-mm", action="store_true")
    audit.add_argument("--block-depth", type=int, default=8)
    audit.add_argument("--limit", type=int)
    audit.add_argument("--output", type=Path, required=True)
    native = sub.add_parser("native-input", help="Verify one TRAIN slab against source CT/GT without resize")
    native.add_argument("--config", type=Path, required=True)
    native.add_argument("--fold", type=int, default=0)
    native.add_argument("--case-id", default="volume-covid19-A-0025")
    native.add_argument("--output", type=Path, required=True)
    profile = sub.add_parser("profile")
    profile.add_argument("--config", type=Path, required=True)
    profile.add_argument("--size", type=int, required=True)
    profile.add_argument("--batch-size", type=int, required=True)
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--warmup", type=int, default=1)
    profile.add_argument("--repeats", type=int, default=2)
    profile.add_argument("--memory-fraction", type=float,
                         help="CUDA allocator budget as a fraction of physical VRAM; use an isolated process")
    profile.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--fold", type=int, default=0)
    pilot.add_argument("--case-id", default="volume-covid19-A-0025")
    pilot.add_argument("--size", type=int, default=128)
    pilot.add_argument("--steps", type=int, default=4)
    pilot.add_argument("--device", default="cuda")
    pilot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        from merit_covid.data.pipeline.preflight import inspect_training
        result = inspect_training(args.config, args.output, fold=args.fold, sizes=args.sizes,
            assume_mm=args.assume_mm, block_depth=args.block_depth, limit=args.limit)
    elif args.command == "native-input":
        from merit_covid.engine.preflight import check_native_input
        result = check_native_input(args.config, args.output, fold=args.fold, case_id=args.case_id)
    else:
        from merit_covid.engine.preflight import profile_model, pilot_case
        if args.command == "profile":
            result = profile_model(args.config, args.output, size=args.size, batch_size=args.batch_size,
                device=args.device, warmup=args.warmup, repeats=args.repeats,
                memory_fraction=args.memory_fraction)
        else:
            result = pilot_case(args.config, args.output, fold=args.fold, case_id=args.case_id,
                size=args.size, steps=args.steps, device=args.device)
    print(f"{args.command}: {result['status']}", flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
