"""The live Kaggle process must import source installed after its startup."""
import base64
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def test_archive_bootstrap_imports_in_current_and_child_process(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('src/merit_covid/__init__.py', 'MARKER = "packaged-source"\n')
    runtime = (Path(__file__).resolve().parents[1] / 'scripts/kaggle/run_training.py').read_text(encoding='utf-8')
    bootstrap = runtime.split('\nimport torch\n', 1)[0]
    code = ('BUNDLE = ' + repr(base64.b64encode(archive.getvalue()).decode()) + '\n'
            + 'SOURCE_SHA = "fixture"\n' + bootstrap + '\n'
            + 'assert merit_covid.MARKER == "packaged-source"\n'
            + 'subprocess.run([sys.executable, "-c", "import merit_covid; assert merit_covid.MARKER == \'packaged-source\'"], check=True)\n')
    env = dict(os.environ, MERIT_SKIP_INSTALL='1', MERIT_SETUP_ONLY='1',
               MERIT_PROJECT_DIR=str(tmp_path / 'source'), PYTHONPATH='')
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
