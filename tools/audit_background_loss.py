"""Analytic segmentation loss for uniform foreground probabilities on train GT.

This is an objective diagnostic, not a measurement of any checkpoint's outputs.
It uses the saved native per-slice lesion areas and the frozen Gaussian/unique
boundary policy, without loading CTs or changing labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def audit(checkpoint, summary, output):
    cp = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    summary = json.loads(Path(summary).read_text())
    ids = set(cp['provenance']['partitions']['train']['case_ids'])
    assert cp['config']['loss']['weighting_mode'] == 'index'
    assert cp['config']['preprocess']['slab_size'] == 11
    assert cp['config']['preprocess']['duplicate_slice_policy'] == 'unique'
    areas = {r['case_id']: np.asarray(r['areas_by_slice_voxels'], dtype=np.float64)
             for r in summary['per_case_slice_areas'] if r['case_id'] in ids}
    assert set(areas) == ids
    pixels = 512*512
    offsets = np.arange(-5, 6)
    gaussian = np.exp(-0.5*(offsets/cp['config']['loss']['sigma_slices'])**2)
    weighted_area, weighted_positive = [], []
    for counts in areas.values():
        for master in range(len(counts)):
            selected = np.clip(master+offsets, 0, len(counts)-1)
            valid = np.zeros(11, dtype=bool)
            seen = set()
            in_bounds = (master+offsets >= 0) & (master+offsets < len(counts))
            for i in sorted(range(11), key=lambda i:(not in_bounds[i], abs(i-5), i)):
                if int(selected[i]) not in seen:
                    valid[i] = True; seen.add(int(selected[i]))
            weights = gaussian*valid
            weighted_area.append((counts[selected], weights/weights.sum()))
            weighted_positive.append(float(((counts[selected]>0)*weights).sum()/weights.sum()))
    results = []
    smooth = cp['config']['loss']['smooth']
    for p in [1e-1, 1e-3, 1e-5, 1e-8, 1e-10, 1e-12, 1e-20]:
        losses = []
        for counts, weights in weighted_area:
            dice = 1-(2*p*counts+smooth)/(p*pixels+counts+smooth)
            fraction = counts/pixels
            ce = -fraction*np.log(p)-(1-fraction)*np.log1p(-p)
            losses.append(float(((dice+ce)*weights).sum()))
        results.append({'constant_foreground_probability': p, 'mean_segmentation_loss': float(np.mean(losses))})
    report = {'scope': __doc__, 'train_cases': len(areas), 'train_masters': len(weighted_area),
        'positive_master_fraction': float(sum(int((v>0).sum()) for v in areas.values())/len(weighted_area)),
        'foreground_voxel_fraction': float(sum(v.sum() for v in areas.values())/(len(weighted_area)*pixels)),
        'weighted_positive_slice_fraction': float(np.mean(weighted_positive)), 'uniform_prediction_loss_curve': results}
    Path(output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--summary', type=Path, default=Path('analysis/covid19_20/summary.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/kaggle_epoch1_quality_audit/background-loss-audit.json'))
    args = parser.parse_args()
    audit(args.checkpoint, args.summary, args.output)
