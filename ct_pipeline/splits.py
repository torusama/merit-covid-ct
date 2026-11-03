"""Deterministic group-disjoint outer test folds with inner validation."""
from __future__ import annotations

from collections import Counter, defaultdict
import random

from .cases import manifest_fingerprint


def create_splits(cases, provenance, seed=42, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    grouped = defaultdict(list)
    for case in cases:
        grouped[case.group_id].append(case.case_id)
    if len(grouped) < 5:
        raise ValueError("Five folds require at least five distinct groups")
    rng = random.Random(seed)
    groups = sorted(grouped)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(grouped[g]))  # Stable seeded tie order.
    buckets, sizes = [[] for _ in range(5)], [0] * 5
    for group in groups:
        i = min(range(5), key=lambda j: sizes[j])
        buckets[i].append(group)
        sizes[i] += len(grouped[group])

    def partition(groups):
        return {"case_ids": sorted(c for g in groups for c in grouped[g]), "group_ids": sorted(groups)}

    folds = []
    for i, test in enumerate(buckets):
        remaining = sorted(set(grouped) - set(test))
        random.Random(seed + 1 + i).shuffle(remaining)
        target = sum(len(grouped[g]) for g in remaining) * validation_fraction
        val, count = [], 0
        for group in remaining[:-1]:
            if val and abs(count - target) <= abs(count + len(grouped[group]) - target):
                continue
            val.append(group)
            count += len(grouped[group])
        train = sorted(set(remaining) - set(val))
        folds.append({"fold": i, "train": partition(train), "validation": partition(val), "test": partition(test)})
    result = {
        "schema_version": 1, "seed": seed, "n_folds": 5,
        "validation_fraction_of_remainder": validation_fraction,
        "manifest_fingerprint": manifest_fingerprint(cases),
        "identity": provenance, "stratification": "none; no trusted lesion-burden statistics supplied",
        "folds": folds,
    }
    validate_splits(result, cases)
    return result


def validate_splits(split, cases):
    by_id = {c.case_id: c.group_id for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate manifest case IDs")
    if split.get("manifest_fingerprint") != manifest_fingerprint(cases):
        raise ValueError("Split does not match the current manifest/group mapping")
    folds = split["folds"]
    if len(folds) != 5 or {f["fold"] for f in folds} != set(range(5)):
        raise ValueError("Expected folds 0 through 4 exactly once")
    test_cases, test_groups = Counter(), Counter()
    for fold in folds:
        seen_cases, seen_groups = set(), set()
        for name in ("train", "validation", "test"):
            part = fold[name]
            ids, groups = part["case_ids"], part["group_ids"]
            if not ids or len(set(ids)) != len(ids) or len(set(groups)) != len(groups):
                raise ValueError(f"Empty or duplicated entries in {name}")
            if not set(ids) <= set(by_id):
                raise ValueError("Unknown case IDs")
            if set(groups) != {by_id[c] for c in ids}:
                raise ValueError("Incorrect group IDs")
            if seen_cases.intersection(ids) or seen_groups.intersection(groups):
                raise ValueError("Case/group leakage between partitions")
            seen_cases.update(ids)
            seen_groups.update(groups)
        if seen_cases != set(by_id):
            raise ValueError("Missing or extra cases")
        test_cases.update(fold["test"]["case_ids"])
        test_groups.update(fold["test"]["group_ids"])
    if test_cases != Counter({c: 1 for c in by_id}) or test_groups != Counter({g: 1 for g in by_id.values()}):
        raise ValueError("Every case and group must occur in exactly one outer test fold")
