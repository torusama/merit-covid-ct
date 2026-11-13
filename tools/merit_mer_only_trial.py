"""One-case experiment: checkpoint MER query blocks only, retaining the model.

The previously validated checkpoint and real-case helpers are reused unchanged.
Apply the candidate only to fresh model instances in an isolated worker.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import types

if __package__:
    from . import merit_checkpoint_trial as memory
    from . import merit_real_case_trial as pilot
else:
    import merit_checkpoint_trial as memory
    import merit_real_case_trial as pilot


def enable_mer_only(model):
    if getattr(model, "_checkpoint_trial_mode", None) == "mer_only":
        return model
    if getattr(model, "_memory_trial_enabled", False):
        raise ValueError("MER-only mode needs a fresh model, not an already patched baseline")
    if model.mer is None:
        raise ValueError("MER-only trial requires MER enabled")
    before = {name: id(value) for name, value in model.state_dict(keep_vars=True).items()}
    model.mer.forward = types.MethodType(memory.checkpointed_mer, model.mer)
    after = {name: id(value) for name, value in model.state_dict(keep_vars=True).items()}
    if before != after:
        raise AssertionError("Execution patch changed model parameter/buffer identities")
    model._checkpoint_trial_mode = "mer_only"
    model._memory_trial_enabled = True
    return model


@contextmanager
def candidate_execution():
    """Gate both numerical comparisons and the pilot on the candidate mode."""
    old_memory, old_pilot = memory.enable_checkpointing, pilot.enable_checkpointing
    memory.enable_checkpointing = pilot.enable_checkpointing = enable_mer_only
    try:
        yield
    finally:
        memory.enable_checkpointing, pilot.enable_checkpointing = old_memory, old_pilot


def run_candidate(*args, **kwargs):
    with candidate_execution():
        report = pilot.run_trial(*args, **kwargs)
    report.update(execution_mode="mer_only", checkpointed_regions=["MER query blocks including distance bias"],
        uncheckpointed_regions=["EPA", "TEP", "encoder1", "final decoder"],
        comparison_scope="one case, two fixed masters; not a cohort speed benchmark")
    output = Path(args[5] if len(args) > 5 else kwargs["output"])
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "ct", "mask", "split", "spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=1.)
    args = parser.parse_args()
    report = run_candidate(args.config, args.ct, args.mask, args.split,
        json.loads(args.spec.read_text()), args.output, steps=args.steps, memory_fraction=args.memory_fraction)
    print("MER-only real pilot:", report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
