"""Build a resumable private Kaggle kernel with the measured native-512 MER.

The source snapshot is immutable and reused for resume. This only prepares
files locally; pushing the kernel is a separate, explicit user action.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/kaggle_factorized_fold0"


def source_snapshot(root=ROOT):
    buffer = io.BytesIO()
    paths = []
    for directory in ("src", "scripts", "configs", "splits"):
        paths.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths.extend(root / name for name in ("pyproject.toml", "data/covid19_20/cases.json"))
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def expected_hashes(root=ROOT):
    result = {}
    for kind in ("ct", "seg"):
        path = root / f"data/covid19_20/Train/volume-covid19-A-0025_{kind}.nii.gz"
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[kind] = digest.hexdigest()
    return result


def runtime_code(snapshot):
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        code = archive.read("scripts/kaggle/run_training.py").decode("utf-8").replace("\r\n", "\n")
        # The source snapshot, not the potentially changed working tree, owns the runner.
        assert archive.read("configs/merit_native512_factorized.yaml")
    edits = [
        ("config = project / 'configs/merit_native512.yaml'",
         "config = project / 'configs/merit_native512_factorized.yaml'"),
        ("'--epochs', '1', '--checkpoint-every', '25'",
         "'--epochs', '100', '--checkpoint-every', '100'"),
        ("pilot = Path('/tmp/merit-pilot')",
         "run([sys.executable, '-u', 'scripts/kaggle/factorized_guard.py', '--output', output / 'factorized-gpu-guard.json'])\npilot = Path('/tmp/merit-pilot')"),
        ("status = 'epoch_complete' if (checkpoint_dir / 'last.pt').is_file() else 'paused_at_checkpoint'",
         "status = 'paused_at_checkpoint' if (checkpoint_dir / 'in_progress.pt').is_file() else 'epoch_complete' if (checkpoint_dir / 'last.pt').is_file() else 'no_checkpoint'"),
        ("'checkpoint': 'checkpoints/last.pt' if status == 'epoch_complete' else\n                  'checkpoints/in_progress.pt' if (checkpoint_dir / 'in_progress.pt').is_file() else None",
         "'checkpoint': 'checkpoints/in_progress.pt' if (checkpoint_dir / 'in_progress.pt').is_file() else 'checkpoints/last.pt' if (checkpoint_dir / 'last.pt').is_file() else None"),
    ]
    for before, after in edits:
        if code.count(before) != 1:
            raise ValueError("Kaggle runner changed; cannot safely adapt expected command/status")
        code = code.replace(before, after, 1)
    return code


def build_kernel(snapshot, *, git_commit, case_hashes, resume_from=None, all_hashes=None, kernel_id=None):
    digest = hashlib.sha256(snapshot).hexdigest()
    prefix = ("BUNDLE = " + repr(base64.b64encode(snapshot).decode()) + "\n"
        + "SOURCE_SHA = " + repr(digest) + "\nGIT_COMMIT = " + repr(git_commit)
        + "\nEXPECTED = " + repr(case_hashes) + "\nRESUME_REQUIRED = " + repr(bool(resume_from))
        + "\nDATA_HASHES = " + repr(all_hashes) + "\nREPAIR_SOURCE_SHA = None\n")
    script = prefix + runtime_code(snapshot)
    compile(script, "kaggle_train.py", "exec")
    slug = kernel_id or ("vtan2405/merit-factorized-fold0-resume" if resume_from
                         else "vtan2405/merit-factorized-fold0")
    if not slug.startswith("vtan2405/") or len(slug.split("/")) != 2 or slug == resume_from:
        raise ValueError("Kernel ID must be a distinct vtan2405/<slug>")
    metadata = dict(id=slug, title="MERIT Factorized Fold0 Resume" if resume_from else "MERIT Factorized Fold0",
        code_file="kaggle_train.py", language="python", kernel_type="script", is_private="true",
        enable_gpu="true", enable_internet="true", machine_shape="NvidiaTeslaT4",
        dataset_sources=["vtan2405/merit-covid19-20"], competition_sources=[],
        kernel_sources=[resume_from] if resume_from else [], model_sources=[])
    return script, metadata, digest


def verify_previous(resume_from, expected_sha):
    kaggle = ROOT / "tools/kaggle-cli/Scripts/kaggle.exe"
    if not kaggle.is_file():
        raise FileNotFoundError("Local Kaggle CLI is missing")
    with tempfile.TemporaryDirectory(prefix="merit-factorized-provenance-") as folder:
        subprocess.run([str(kaggle), "kernels", "output", resume_from, "-p", folder,
                        "--file-pattern", r"source_provenance\.json$"], check=True)
        reports = list(Path(folder).rglob("source_provenance.json"))
        if len(reports) != 1:
            raise ValueError("Previous source provenance missing or ambiguous")
        previous = json.loads(reports[0].read_text(encoding="utf-8"))
        if previous.get("archive_sha256") != expected_sha:
            raise ValueError("Previous run used a different source snapshot; refusing resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", help="Saved prior kernel with in_progress.pt")
    parser.add_argument("--kernel-id", help="Distinct vtan2405/<slug> for this Kaggle version")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output / "source_snapshot.zip"
    provenance_path = args.output / "package_provenance.json"
    if args.resume_from:
        if not snapshot_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError("Resume needs the original local source_snapshot.zip and package_provenance.json")
        snapshot = snapshot_path.read_bytes()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(snapshot).hexdigest()
        if digest != provenance["source_sha256"]:
            raise ValueError("Local saved snapshot hash mismatch")
        verify_previous(args.resume_from, digest)
        commit, hashes = provenance["git_commit"], provenance["a0025_uncompressed_sha256"]
        all_hashes = json.loads((ROOT / "outputs/kaggle_results/source-uncompressed-hashes.json").read_text(encoding="utf-8"))
    else:
        snapshot = source_snapshot()
        if snapshot_path.exists() and snapshot_path.read_bytes() != snapshot:
            raise FileExistsError("Existing source snapshot differs; select a new --output directory")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        hashes, all_hashes = expected_hashes(), None
    script, metadata, digest = build_kernel(snapshot, git_commit=commit, case_hashes=hashes,
        resume_from=args.resume_from, all_hashes=all_hashes, kernel_id=args.kernel_id)
    snapshot_path.write_bytes(snapshot)
    if not args.resume_from:
        provenance_path.write_text(json.dumps(dict(source_sha256=digest, git_commit=commit,
            a0025_uncompressed_sha256=hashes, training_config="configs/merit_native512_factorized.yaml",
            checkpoint_interval_batches=100, epoch_budget=100), indent=2) + "\n", encoding="utf-8")
    (args.output / "kaggle_train.py").write_text(script, encoding="utf-8")
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Prepared", args.output, "source SHA256", digest, "kernel", metadata["id"])


if __name__ == "__main__":
    main()
