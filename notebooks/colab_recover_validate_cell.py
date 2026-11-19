#@title Recover saved checkpoints and resume validation (no training)
import torch
"""Recover existing paired-run snapshots without changing the frozen worker."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import zipfile


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inspect(path, arm, binding, schema, batches, signature):
    import torch
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError('ZIP CRC failed: ' + bad)
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    try:
        if saved.get('schema_version') != schema:
            raise ValueError('Checkpoint schema differs')
        if saved.get('input_binding') != binding:
            raise ValueError('Source, parent or dataset identity differs')
        report = saved['report']
        if report.get('arm') != arm:
            raise ValueError('Checkpoint belongs to another arm')
        if report.get('processed_batches') != batches or report.get('requested_batches') != batches:
            raise ValueError(f"Checkpoint has {report.get('processed_batches')} / {batches} batches")
        if signature is not None and saved.get('signature') != signature:
            raise ValueError('Checkpoint signature differs from arm-report.json')
        if 'model' not in saved:
            raise ValueError('Missing model state')
    finally:
        del saved
        gc.collect()


def recover_checkpoints(context, output, *, expected_binding, schema,
                        total_batches=300, persist_recovery=True):
    """Return verified local paths; preserve every original checkpoint.

    CRC checks read all tensor bytes. Drive candidates are copied locally before
    inspection. A recovered copy is written under a content-derived new name,
    then read back and hashed; this verifies mounted bytes, not cloud sync state.
    """
    work = Path(context['work'])
    work.mkdir(parents=True, exist_ok=True)
    output = Path(output)
    recovered, failures, records = {}, {}, {}
    for arm in ('reference_alignment', 'alignment_off'):
        destination = output / arm
        progress = destination / 'progress.pt'
        signature = None
        report_path = destination / 'arm-report.json'
        if report_path.is_file():
            signature = json.loads(report_path.read_text())['signature']
        local = [work / (arm + '-save.pt')]
        local += sorted(work.glob(arm + '-recovery-*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        local += sorted(work.glob(arm + '-read-*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        remote = [progress, destination / 'progress.pt.pending']
        remote += sorted(destination.glob('progress-recovered-*.pt'))
        candidates = [(path, False) for path in local] + [(path, True) for path in remote]
        problems = []
        chosen = None
        for candidate, copy_local in candidates:
            if not candidate.is_file():
                continue
            print('CHECK', arm, str(candidate), candidate.stat().st_size, 'bytes', flush=True)
            checked = work / (arm + '-recovery-' + uuid.uuid4().hex + '.pt') if copy_local else candidate
            try:
                if copy_local:
                    shutil.copyfile(candidate, checked)
                    if checked.stat().st_size != candidate.stat().st_size:
                        raise ValueError('Local copy size differs from source')
                _inspect(checked, arm, expected_binding, schema, total_batches, signature)
                chosen = checked
                print('VERIFIED', arm, total_batches, 'batches; complete ZIP and matching input identity', flush=True)
                break
            except (OSError, RuntimeError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                problems.append(dict(path=str(candidate), error=str(error)))
                print('UNUSABLE', str(candidate), str(error), flush=True)
                if copy_local and checked.is_file():
                    checked.unlink()
        if chosen is None:
            failures[arm] = problems or [dict(error='No saved checkpoint files found')]
            print('NO VALID COMPLETE CHECKPOINT', arm, flush=True)
            continue
        recovered[str(progress.resolve())] = str(chosen)
        records[arm] = dict(local=str(chosen), original_progress=str(progress), rejected=problems)
        if persist_recovery:
            digest = _sha256(chosen)
            backup = destination / ('progress-recovered-' + digest[:16] + '.pt')
            try:
                if backup.is_file():
                    if _sha256(backup) != digest:
                        raise ValueError('Existing recovered copy has unexpected content; preserving it')
                else:
                    pending = backup.with_name(backup.name + '.' + uuid.uuid4().hex + '.pending')
                    try:
                        shutil.copyfile(chosen, pending)
                        if _sha256(pending) != digest:
                            raise ValueError('Drive recovery copy hash mismatch')
                        pending.rename(backup)
                    finally:
                        if pending.exists():
                            pending.unlink()
                records[arm].update(recovery_copy=str(backup), sha256=digest)
                print('RECOVERY COPY VERIFIED', str(backup), flush=True)
            except (OSError, ValueError) as error:
                records[arm]['recovery_copy_error'] = str(error)
                print('WARNING: local checkpoint is valid; recovery copy could not be saved:', str(error), flush=True)
        records[arm]['processed_batches'] = total_batches
    receipt = dict(recovered=records, failures=failures)
    (work / 'checkpoint-recovery-review.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return recovered, receipt


def install_recovered_reader(validate_function, recovered):
    """Redirect only this run's progress paths; keep all worker checks intact."""
    namespace = validate_function.__globals__
    previous = namespace['local_snapshot']

    def reader(context, path):
        verified = recovered.get(str(Path(path).resolve()))
        return Path(verified) if verified is not None else previous(context, path)

    namespace['local_snapshot'] = reader

# Use in one new cell of the ORIGINAL prepared Colab runtime.
if "context" not in globals() or "validate_arm" not in globals():
    raise RuntimeError("Run original notebook Cells 1-7 with the same RUN_NAME first; do not rerun training.")
from tools.order_optimization_pilot import state_hash
import time
expected_binding = state_hash({key: context[key] for key in ("parent_sha256", "source_sha256", "hashes")})
recovered_paths, recovery_review = recover_checkpoints(
    context, OUTPUT_DIR, expected_binding=expected_binding,
    schema=validate_arm.__globals__["SCHEMA"], total_batches=TOTAL_BATCHES,
)
if not recovered_paths:
    raise RuntimeError("No intact matching 300-batch checkpoint found. Keep the recovery output; do not overwrite or retrain yet.")
install_recovered_reader(validate_arm, recovered_paths)
SESSION_STARTED = time.monotonic()
RUN_FULL_VALIDATION = True
print("Validation budget restarted. Original checkpoints are preserved.")
for recovery_arm in ("reference_alignment", "alignment_off"):
    recovery_progress = Path(OUTPUT_DIR) / recovery_arm / "progress.pt"
    if str(recovery_progress.resolve()) not in recovered_paths:
        print("SKIP: no verified complete checkpoint for", recovery_arm)
        continue
    if remaining_seconds() <= 0:
        print("Validation budget exhausted. Rerun this recovery cell to resume saved cases.")
        break
    recovery_result = validate_arm(context, OUTPUT_DIR, recovery_arm, max_seconds=remaining_seconds())
    print(recovery_arm, recovery_result.get("status", "validation_complete"))
    gc.collect()
    torch.cuda.empty_cache()
recovery_comparison = compare(OUTPUT_DIR)
print("Comparison status:", recovery_comparison["status"])
print("Recovery details:", Path(context["work"]) / "checkpoint-recovery-review.json")
