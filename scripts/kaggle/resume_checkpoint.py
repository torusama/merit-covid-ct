"""Verify dataset bytes before rebinding Kaggle-only source timestamps."""
from copy import deepcopy
import gzip
import hashlib


def verify_dataset_content(cases, expected):
    if {c.case_id for c in cases} != set(expected):
        raise ValueError('Dataset hash manifest case IDs differ')
    for index, case in enumerate(cases):
        for key, path in [('image', case.image), ('mask', case.mask)]:
            digest = hashlib.sha256()
            opener = gzip.open if str(path).endswith('.gz') else open
            with opener(path, 'rb') as handle:
                for block in iter(lambda: handle.read(65536), b''):
                    digest.update(block)
            if digest.hexdigest() != expected[case.case_id][key]:
                raise ValueError(f'Dataset bytes differ: {case.case_id} {key}')
        if (index + 1) % 25 == 0:
            print(f'Content verified: {index+1}/{len(cases)} CT/mask pairs', flush=True)


def rebind_verified_timestamps(checkpoint, config, provenance, fingerprint):
    from merit_covid.engine.training import protocol_fingerprint
    old = checkpoint['provenance']
    if checkpoint['protocol_fingerprint'] != protocol_fingerprint(checkpoint['config'], old):
        raise ValueError('Checkpoint protocol fingerprint is internally inconsistent')
    def without_timestamps(value):
        result = deepcopy(value)
        for row in result['source_identity']['cases']:
            for key in ('image', 'mask'):
                row[key].pop('mtime_ns', None)
        return result
    if checkpoint['config'] != config or without_timestamps(old) != without_timestamps(provenance):
        raise ValueError('Resume differs by more than source timestamps; refusing migration')
    result = checkpoint.copy()
    result['provenance'] = deepcopy(provenance)
    result['protocol_fingerprint'] = fingerprint
    result['verified_timestamp_migration'] = {'previous_fingerprint':checkpoint['protocol_fingerprint'],
                                            'verification':'all CT/mask uncompressed SHA256 matched local source'}
    return result
