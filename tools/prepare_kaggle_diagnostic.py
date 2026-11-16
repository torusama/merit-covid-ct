"""Reuse the last exact source archive, attach its output, replay the failing interval."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root/'outputs/kaggle_native512_fold0/kaggle_train.py').read_text(encoding='utf-8')
prefix = source.split('\ndef run(args):',1)[0]
import base64, io, zipfile
# Add the diagnostic-only migration helper to the frozen source snapshot.
import ast
tree = ast.parse(source)
encoded = ast.literal_eval(tree.body[0].value)
archive = io.BytesIO(base64.b64decode(encoded))
with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(root/'scripts/kaggle/resume_checkpoint.py','scripts/kaggle/resume_checkpoint.py')
prefix = prefix.replace(repr(encoded), repr(base64.b64encode(archive.getvalue()).decode()),1)
hashes=json.loads((root/'outputs/kaggle_results/source-uncompressed-hashes.json').read_text())
prefix = 'DATA_HASHES = ' + repr(hashes) + '\n' + prefix
diagnosis = (root/'scripts/kaggle/diagnose_resume.py').read_text(encoding='utf-8')
dest = root/'outputs/kaggle_diagnostic'
dest.mkdir(parents=True,exist_ok=True)
code = prefix + '\n' + diagnosis
compile(code,'diagnose.py','exec')
(dest/'diagnose.py').write_text(code,encoding='utf-8')
metadata = json.loads((root/'outputs/kaggle_native512_fold0/kernel-metadata.json').read_text())
metadata.update(id='vtan2405/merit-native512-resume-diagnostic', title='MERIT Native512 Resume Diagnostic',
                code_file='diagnose.py', kernel_sources=['vtan2405/merit-native512-fold0'])
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(dest)
