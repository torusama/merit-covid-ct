"""Compare MER query chunk sizes on the existing real-CT diagnostic.

This is an isolated experiment: it does not change the production config or
claim accuracy, and each size must run in a fresh process.
"""
import argparse
import json
from pathlib import Path

from merit_covid.engine import activation_checkpointing as execution
from tools.merit_attention_pilot import run_pilot
from tools.merit_factorized_attention import enable_factorized_mer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-dir", "specs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--chunk", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    original = execution.configure_model_execution

    def configure(model, settings):
        model = enable_factorized_mer(original(model, settings))
        model.mer.query_chunk_size = args.chunk
        return model

    execution.configure_model_execution = configure
    try:
        report = run_pilot(args.config, args.data_dir,
                           json.loads(args.specs.read_text(encoding="utf-8")),
                           args.output, variant="optimized", rounds=args.rounds)
    finally:
        execution.configure_model_execution = original
    print("Chunk pilot:", args.chunk, report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
