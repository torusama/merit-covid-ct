"""Private factorized kernel packaging: frozen source identity and resume."""
import io
from pathlib import Path
import zipfile

import pytest

from tools.prepare_kaggle_factorized import build_kernel, runtime_code, source_snapshot


def test_factorized_kernel_contains_frozen_config_and_mid_epoch_status():
    root = Path(__file__).resolve().parents[1]
    snapshot = source_snapshot(root)
    assert snapshot == source_snapshot(root)
    script, metadata, _ = build_kernel(snapshot, git_commit="fixture", case_hashes={"ct": "a", "seg": "b"})
    compile(script, "kaggle_train.py", "exec")
    assert "configs/merit_native512_factorized.yaml" in script
    assert "'--epochs', '100', '--checkpoint-every', '100'" in script
    assert script.index("scripts/kaggle/factorized_guard.py") < script.index("scripts/merit_preflight.py")
    assert "status = 'paused_at_checkpoint' if (checkpoint_dir / 'in_progress.pt').is_file()" in script
    assert metadata["id"] == "vtan2405/merit-factorized-fold0"
    assert metadata["enable_gpu"] == "true" and not metadata["kernel_sources"]
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        assert archive.read("src/merit_covid/engine/factorized_attention.py") == (
            root / "tools/merit_factorized_attention.py").read_bytes()
        assert b"factorized_mer: true" in archive.read("configs/merit_native512_factorized.yaml")
        assert not any(name.endswith((".nii", ".nii.gz", ".pt")) for name in archive.namelist())


def test_resume_source_and_kernel_identity_are_explicit():
    snapshot = source_snapshot()
    source = "vtan2405/merit-factorized-fold0"
    script, metadata, _ = build_kernel(snapshot, git_commit="fixture", case_hashes={},
        resume_from=source, all_hashes={"a": {"image": "hash", "mask": "hash"}})
    assert metadata["kernel_sources"] == [source]
    assert metadata["id"] != source
    assert "RESUME_REQUIRED = True" in script
    assert "--verified-resume-hashes" in script
    with pytest.raises(ValueError, match="distinct"):
        build_kernel(snapshot, git_commit="fixture", case_hashes={}, resume_from=source, kernel_id=source)


def test_changed_runner_is_not_silently_rewritten():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scripts/kaggle/run_training.py", "print('unexpected runner')")
        archive.writestr("configs/merit_native512_factorized.yaml", "factorized_mer: true")
    with pytest.raises(ValueError, match="runner changed"):
        runtime_code(buffer.getvalue())
