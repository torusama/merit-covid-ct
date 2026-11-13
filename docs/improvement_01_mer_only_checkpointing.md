# Improvement note 01: MER-only checkpointing

Date: 2026-11-13. Status: **prototype notebook ready; A-0025/T4 measurement pending**.
This note defines one improvement experiment on one CT volume. It does not
authorize running the complete improvement plan or the full cohort.

## Goal

Try to reduce training time by retaining more backbone activations in the T4's
available VRAM, while preserving the full native input and MERIT design.
The current real-case pilot uses substantially less memory than the assigned
GPU provides, so less recomputation is worth measuring. No speedup is assumed.

## One-case baseline

Evidence: user-supplied `20261113_081310_7f8b19a4.zip`, summarized in
[validation.md](validation.md). Existing workflow:
[colab_real_case.ipynb](../notebooks/colab_real_case.ipynb).

| Setting/result | Recorded baseline |
| --- | --- |
| Case | volume-covid19-A-0025, fold 0 TRAIN |
| Scope | One volume; masters 30 and 31; two optimizer attempts |
| Input | `[1,1,11,512,512]`, native CT/GT, no resize or intensity normalization |
| Model | Feature size 16, depths 3/3/3/3, TEP/MER/PCG enabled; 174,043,602 parameters |
| Hardware | Tesla T4; runtime reports 14.56 GiB total VRAM |
| Execution | MER query-block checkpointing plus EPA, TEP, encoder1 and final decoder checkpointing |
| Updates | 2 successful AdamW/AMP updates; no AMP skips |
| Step times | 13.958 s and 13.731 s |
| Peak PyTorch memory | 4649.68 MiB allocated; 5072 MiB reserved |

## Only experimental change

The isolated [MER-only helper](../tools/merit_mer_only_trial.py) reuses the
validated checkpoint and real-case helpers unchanged. Run
[colab_mer_only_case.ipynb](../notebooks/colab_mer_only_case.ipynb) with T4 and the
same `A0025_CT_GT.zip`, then return its report ZIP. No config edits are needed.

- Keep checkpointing each MER query block, including construction of its distance bias.
- Run EPA, TEP, encoder1 and the final decoder normally, without checkpointing them.
- Keep the current all-region checkpoint mode available as the measured baseline.
- Leave production model files and default trainer behavior unchanged at this stage.

Preserve model/state shapes, all attention context keys, query chunk size 128,
loss, optimizer, AMP/scaler policy, seed 42, epoch-0 PCG warmup, source hashes,
sampling and the frozen split. Do not resize, alter labels, remove modules or
reduce channels/layers to make this experiment pass.

## Next execution steps

1. Compare the candidate against the original model on compact fixtures:
   outputs, losses, input/all parameter gradients, state keys/counts and BatchNorm
   buffers must agree within the existing recorded tolerances.
2. If comparison passes, run only the same A-0025 pilot on T4: masters 30/31,
   batch 1, native 512, two optimizer attempts, allocator fraction 1.0.
3. Record both step times, successful/skipped updates, finite gradient checks,
   peak allocated/reserved memory and source-file integrity in a small report ZIP.
4. Compare with the baseline and stop. If it OOMs or is not clearly faster,
   retain the baseline and report the outcome; do not silently change other settings.

## Result to fill after the experiment

| Candidate result | Value |
| --- | --- |
| Numerical/buffer comparison | Compact CPU/GPU fixtures PASS; repeated on the assigned GPU before the real pilot |
| Successful optimizer updates | NOT RUN |
| Step times and memory peaks | NOT RUN |
| Decision and limitations | NOT RUN |

The complete notebook workflow was checked locally with synthetic native 32x32
NIfTI: the candidate completed an actual GPU update, passed the CT-only guard,
and exported the report. This is not a native-512 A-0025 timing measurement.
`comparison.json` includes reference metrics, candidate results and environment/
initial-weight matching flags. Relative time changes are calculated only when
hardware/Torch and initial parameter hashes match the reference; two updates
still do not establish a reliable speedup.

Two steps on one case are an initial comparison, not a reliable cohort-throughput
or segmentation-quality result. Repeat the same case if timing noise prevents a
decision. Multi-case testing, production integration, mid-epoch resume and full
fold training are separate later tasks, outside this note's scope.
