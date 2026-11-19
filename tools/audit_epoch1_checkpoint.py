"""Read-only comparison of the frozen epoch-0/epoch-1 checkpoints."""
import argparse
import bisect
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def audit(old_path, new_path, manifest_path, output):
    torch.set_num_threads(2)
    old = torch.load(old_path, map_location='cpu', mmap=True, weights_only=False)
    new = torch.load(new_path, map_location='cpu', mmap=True, weights_only=False)
    report = {'epochs': [old['epoch'], new['epoch']], 'nonfinite': {}, 'batchnorm': [],
              'scalar_changes': {}, 'config_changes': {}}
    for section in old['config']:
        for key in old['config'][section].keys() | new['config'][section].keys():
            a, b = old['config'][section].get(key), new['config'][section].get(key)
            if a != b:
                report['config_changes'][section + '.' + key] = [a, b]
    for label, cp in [('epoch0', old), ('epoch1', new)]:
        report['nonfinite'][label] = [name for name, value in cp['model'].items()
                                    if not torch.isfinite(value).all()]
    for name, a in old['model'].items():
        b = new['model'][name]
        if a.numel() == 1 and 'num_batches_tracked' not in name:
            report['scalar_changes'][name] = [a.item(), b.item()]
        if name.endswith('running_var'):
            mean_name = name.replace('running_var', 'running_mean')
            count_name = name.replace('running_var', 'num_batches_tracked')
            ratio = b / a.clamp_min(1e-12)
            shift = (new['model'][mean_name] - old['model'][mean_name]).abs() / a.sqrt().clamp_min(1e-6)
            report['batchnorm'].append({'name': name, 'old_variance_max': a.max().item(),
                'new_variance_max': b.max().item(), 'variance_ratio_min': ratio.min().item(),
                'variance_ratio_max': ratio.max().item(), 'mean_shift_old_std_max': shift.max().item(),
                'batches_tracked': [old['model'][count_name].item(), new['model'][count_name].item()]})
    records = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    selected = set(old['provenance']['partitions']['train']['case_ids'])
    records = [r for r in records if r['id'] in selected]
    ends = [0]
    for r in records:
        ends.append(ends[-1] + r['shape'][2])
    generator = torch.Generator()
    generator.set_state(old['loader_generator'])
    # DataLoader iterator consumes one base seed before the sampler starts.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = torch.randperm(len(records), generator=generator).tolist()
    indices = []
    for i in order:
        indices.extend(ends[i] + j for j in torch.randperm(ends[i+1] - ends[i], generator=generator).tolist())
    report['epoch1_order'] = {'batches': len(indices),
        'final_generator_matches_checkpoint': torch.equal(generator.get_state(), new['loader_generator']),
        'last_cases': [records[i]['id'] for i in order[-5:]], 'last_100': []}
    mask_cache = {}
    for index in indices[-100:]:
        i = bisect.bisect_right(ends, index) - 1
        record = records[i]
        if i not in mask_cache:
            image = nib.as_closest_canonical(nib.load(str(Path(manifest_path).parent / record['mask'])))
            mask_cache[i] = np.asarray(image.dataobj) > 0
        z = index - ends[i]
        mask = mask_cache[i]
        lo, hi = max(0, z-5), min(mask.shape[2], z+6)
        report['epoch1_order']['last_100'].append({'case': record['id'], 'master': z,
            'master_positive_voxels': int(mask[:, :, z].sum()), 'slab_has_lesion': bool(mask[:, :, lo:hi].any())})
    rows = report['epoch1_order']['last_100']
    report['epoch1_order']['last_100_empty_masters'] = sum(r['master_positive_voxels'] == 0 for r in rows)
    report['epoch1_order']['last_100_empty_slabs'] = sum(not r['slab_has_lesion'] for r in rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('batchnorm', 'epoch1_order')}, indent=2))
    print('Worst BN variance ratios:', sorted(report['batchnorm'], key=lambda r:r['variance_ratio_max'], reverse=True)[:4])
    print('Sampling:', {k: v for k, v in report['epoch1_order'].items() if k != 'last_100'})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('epoch0', type=Path)
    parser.add_argument('epoch1', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('data/covid19_20/cases.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/kaggle_epoch1_quality_audit/checkpoint-audit.json'))
    args = parser.parse_args()
    audit(args.epoch0, args.epoch1, args.manifest, args.output)
