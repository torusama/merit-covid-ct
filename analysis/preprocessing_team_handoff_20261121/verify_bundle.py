"""Verify the extracted handoff's file hashes using only the Python stdlib."""
import hashlib
import argparse
import csv
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,help='Dataset parent containing Train/, as recorded in the provided manifest')
args=parser.parse_args()
spec=json.loads((root/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
failures=[]
for name,expected in spec.items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(name+': missing or escaping path')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        failures.append(name+': SHA256 mismatch')
dataset=None
if args.data_root is not None and not failures:
    records=json.loads((root/'project/data/covid19_20/cases.json').read_text(encoding='utf-8-sig'))
    with (root/'reports/all199_ct_gt_qc.csv').open(newline='',encoding='utf-8') as f:
        rows={r['case_id']:r for r in csv.DictReader(f)}
    if set(rows)!={r['id'] for r in records} or len(rows)!=199:
        failures.append('Dataset/QC identity differs')
    checked=0
    for record in records:
        for key,prefix in (('image','ct'),('mask','gt')):
            path=args.data_root/record[key]
            if not path.is_file():
                failures.append(str(path)+': missing dataset file')
                continue
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=rows[record['id']][prefix+'_sha256_before']:
                failures.append(str(path)+': differs from the QC199 source')
            checked+=1
    dataset=dict(files_checked=checked,expected_files=398,status='PASS' if not failures and checked==398 else 'FAIL')
print(json.dumps(dict(status='PASS' if not failures else 'FAIL',files=len(spec),dataset=dataset,failures=failures),indent=2))
sys.exit(0 if not failures else 2)
