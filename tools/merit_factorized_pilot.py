"""Benchmark factorized MER on the same diagnostic as the original pilot."""
import argparse
import json
from pathlib import Path

from merit_covid.engine import activation_checkpointing as execution
from tools.merit_factorized_attention import enable_factorized_mer
from tools.merit_attention_pilot import run_pilot


def run_factorized(config, data_dir, specs, output, *, device="cuda", rounds=1):
    original = execution.configure_model_execution
    def candidate(model, settings):
        return enable_factorized_mer(original(model, settings))
    execution.configure_model_execution = candidate
    try:
        return run_pilot(config, data_dir, specs, output, variant="optimized", device=device, rounds=rounds)
    finally:
        execution.configure_model_execution = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data-dir", "specs", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    result = run_factorized(args.config, args.data_dir, json.loads(args.specs.read_text()), args.output, rounds=args.rounds)
    print("Factorized pilot:", result["status"], flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
