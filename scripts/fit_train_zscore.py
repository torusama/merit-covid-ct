"""One CPU-fit entrypoint for frozen TRAIN-only intensity policies.

V3 writes stats/audit for the canonical config. The historical global-zscore
mode still writes its original experiment YAML. No GT or held-out voxels are
opened for fitting, and source NIfTI/manifest/split files are never modified.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import yaml

from merit_covid.engine.training import load_experiment, load_development_split


def fit_train_zscore(config, fold, block_depth=8):
    if type(fold) is not int or fold not in range(5) or type(block_depth) is not int or block_depth < 1:
        raise ValueError("fold must be 0..4 and block_depth positive")
    if config["preprocess"]["intensity_encoding"] != "native":
        raise ValueError("Fit requires a native-intensity source configuration")
    cases, split, _, split_hash = load_development_split(config)
    selected = set(next(f for f in split["folds"] if f["fold"] == fold)["train"]["case_ids"])
    count, mean, m2 = 0, 0.0, 0.0
    for case in cases:
        if case.case_id not in selected:
            continue
        image = nib.load(str(case.image))
        if len(image.shape) != 3:
            raise ValueError(f"{case.case_id}: expected 3D CT")
        for z in range(0, image.shape[2], block_depth):
            block = np.asarray(image.dataobj[:, :, z:z + block_depth], dtype=np.float32)
            if not block.size or not np.isfinite(block).all():
                raise ValueError(f"{case.case_id}: non-finite or empty CT")
            n = block.size
            block_mean = float(np.mean(block, dtype=np.float64))
            block_m2 = float(np.var(block, dtype=np.float64)) * n
            total = count + n
            delta = block_mean - mean
            m2 += block_m2 + delta * delta * count * n / total
            mean += delta * n / total
            count = total
    if not count or not np.isfinite(m2) or m2 / count <= 1e-16:
        raise ValueError("Training voxels have invalid or near-zero variance")
    return {"zscore_mean": mean, "zscore_std": float(np.sqrt(m2 / count)),
            "zscore_fit": {"fold": fold, "split_sha256_lf": split_hash,
                           "case_ids": sorted(selected), "method": "native_scaled_voxel_population"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, help='Historical global train_zscore experiment YAML')
    parser.add_argument('--policy', choices=('train_zscore', 'train_fovaware_clipped_zscore_v1'), default='train_zscore')
    parser.add_argument('--output-dir', type=Path, help='V3 stats/audit directory; existing stats are never overwritten')
    parser.add_argument('--block-depth', type=int, default=4)
    parser.add_argument('--max-volume-mib', type=int, default=768)
    args = parser.parse_args()
    config = load_experiment(args.config)
    if args.policy == 'train_fovaware_clipped_zscore_v1':
        from merit_covid.data.pipeline.intensity_fit import fit_saved_fold
        if args.output_dir is None or args.output is not None:
            parser.error('V3 requires --output-dir (not --output)')
        target = args.output_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        if any((target/name).exists() for name in ('fit_stats.json','fit_status.json','case_padding_audit.csv')):
            raise FileExistsError('Preserve existing stats/audit; choose a fresh output directory')
        status_path = target/'fit_status.json'
        def status(value):
            status_path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        status(dict(status='NOT_FITTED', policy=args.policy, reason='CPU fit in progress; no usable stats yet'))
        try:
            stats, rows = fit_saved_fold(config, fold=args.fold, block_depth=args.block_depth,
                                         max_volume_mib=args.max_volume_mib)
            keys = sorted({key for row in rows for key in row})
            with (target/'case_padding_audit.csv').open('x',newline='',encoding='utf-8') as handle:
                writer = csv.DictWriter(handle,fieldnames=keys)
                writer.writeheader(); writer.writerows(rows)
            if stats['status'] == 'FITTED':
                with (target/'fit_stats.json').open('x',encoding='utf-8') as handle:
                    json.dump(stats,handle,indent=2,allow_nan=False);handle.write('\n')
                status(dict(status='FITTED',policy=args.policy,stats_sha256=stats['stats_sha256'],
                            parameters=stats['parameters'],elapsed_seconds=stats['elapsed_seconds']))
                print(json.dumps(dict(status='FITTED',parameters=stats['parameters'],padding_counts=stats['padding_counts'],
                                      voxel_counts=stats['voxel_counts'],elapsed_seconds=stats['elapsed_seconds']),indent=2))
            else:
                status(stats)
                raise RuntimeError('V3 fit BLOCKED: '+stats['reason'])
        except BaseException as error:
            if not (target/'fit_stats.json').exists() and json.loads(status_path.read_text())['status'] != 'BLOCKED':
                status(dict(status='BLOCKED',policy=args.policy,reason=str(error)))
            raise
        return
    if args.output is None or args.output_dir is not None:
        parser.error('Historical train_zscore requires --output YAML')
    fit = fit_train_zscore(config, args.fold)
    config["preprocess"].update(intensity_encoding="train_zscore", **fit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()
