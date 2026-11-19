# Colab: alignment-only paired continuation

Open [colab_alignment_pair.ipynb](../notebooks/colab_alignment_pair.ipynb) in Colab using **File → Upload notebook**. Source is embedded in the notebook; no repository clone, Kaggle authentication or Kaggle GPU quota is needed.

The notebook has **12 numbered runnable code cells**, each with its own run button: configuration, Drive mount, GPU/input checks, source unpacking, dependencies, dataset staging, checkpoint verification, training A, training B, validation A, validation B and comparison. Run Cell 1 through Cell 12 in order. The large embedded-source cell is collapsed by default.

If an IDE shows raw JSON, reopen the `.ipynb` with its Notebook/Jupyter editor, or upload the file directly into Colab. A [Python percent-cell export](../notebooks/colab_alignment_pair_cells.py) is also provided: each `# %%` block is a separate code cell for a Jupyter-capable editor or manual copying into Colab. Do not paste notebook JSON into a code cell.

## Inputs

Put the following under Google Drive, or edit the three input paths in the first configuration cell:

```text
MyDrive/MERIT_alignment_test/inputs/
  Train/
    volume-..._ct.nii.gz
    volume-..._seg.nii.gz
  epoch0/
    last.pt
    history.jsonl
```

`DRIVE_ROOT` is `/content/drive/MyDrive/MERIT_alignment_test`, with `MERIT_alignment_test` directly inside the mounted My Drive. `DATASET_SOURCE` defaults to `DRIVE_ROOT + "/inputs/Train"`; checkpoint and history default to `inputs/epoch0/last.pt` and `inputs/epoch0/history.jsonl`. Cell 2 preserves an existing Drive mount without invoking Drive authentication again. `FORCE_REMOUNT_DRIVE=False` is the default; set it to `True` only when intentionally remounting. If no Drive is mounted, Cell 2 starts authorization.

If the runtime already has direct access to the folder owner's Drive, use that mount: no shortcut or sharing with the compute account is needed merely to use files already authorized through that mount. Account selection depends on the runtime/extension authorization flow; the notebook does not select an account from a folder URL. If direct owner authentication is unavailable and you instead mount the new account's Drive, sharing the project as **Editor** and adding a **Shared with me → Organize → Add shortcut → My Drive** shortcut is an alternative. Keep that shortcut named `MERIT_alignment_test`, or configure its actual path. See [Google's shortcut instructions](https://support.google.com/drive/answer/9700156?hl=en).

Cell 1 contains the supplied `SHARED_FOLDER_URL`. This identifies the intended folder; a URL alone does not grant access or mount it. `SHARED_PROJECT_PATH` can override the mounted project path, whether it is a direct owner folder or a shortcut. Cell 2 resolves this folder, requires its `inputs` directory, and rebases default data/checkpoint/history paths while preserving explicitly external paths. Outputs continue under the selected project's `runs`, with the same `RUN_NAME`. If the folder is inaccessible, the notebook stops with account/path guidance and the optional sharing route instead of creating an empty replacement project. The frozen training bundle and recovery protocol are unchanged.

The dataset path also accepts a ZIP or a directory containing the NIfTI files. Cell 3 uses configured checkpoint/history paths when they exist; otherwise it searches under `inputs` and accepts only a unique `last.pt` and `history.jsonl`. Missing or ambiguous files produce specific instructions instead of silently choosing a checkpoint. The resolver accepts the original and Kaggle-extracted NIfTI naming conventions. All 199 decompressed CT/mask byte hashes must match the frozen manifest. Directory staging copies only NIfTI files into the runtime's local disk, excluding checkpoint/history files stored alongside the dataset.

Dataset discovery searches CT/seg filenames recursively, so arbitrary wrapper names and separate CT/mask directories are supported; there is no requirement for a folder named `Train`. Verification prints the discovered NIfTI count and `CT=x/199`, `GT=y/199`. Missing files are listed by case/kind; duplicate copies are rejected rather than silently mixed. Staging checks a nested-file inventory before reusing its cache, so a partially completed upload can be refreshed after the remaining files arrive. Cell 5 explicitly reloads the worker after source extraction to avoid using a cached older version of the resolver.

The trusted parent is available locally at:

```text
outputs/kaggle_review_20261116/epoch0_complete/MERIT/fold-00/checkpoints/last.pt
outputs/kaggle_review_20261116/epoch0_complete/MERIT/fold-00/checkpoints/history.jsonl
```

The parent SHA256 is `22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934`. Do not substitute the failed epoch-1 or pilot-300 model. The notebook explicitly binds new file locations after content checks, preserving the original split/group identity and immutable training settings. Unused saved CUDA RNG streams are trimmed when moving from the original two-device Kaggle runtime to one-device Colab; no exact historical cross-runtime replay is claimed.

## Fixed experiment

Select a GPU runtime with approximately 16 GB VRAM or more. Inputs remain native 11 × 512 × 512 slabs. Both branches use fast attention, frozen BN statistics, LR 3e-5, the epoch-1 PCG ramp, inherited optimizer moments and an identical 300-master plan: ten unique masters from each of thirty seeded training volumes, interleaved within groups of five volumes.

| Arm | Alignment weight | Presence/salience losses |
| --- | --- | --- |
| `reference_alignment` | Original 0.05 | Unchanged |
| `alignment_off` | 0 | Unchanged |

This is an alignment-only ablation, not an established repair. Compare the endpoint on all 32 complete native validation volumes, at fixed threshold 0.5. Selected-slice probes are diagnostics. The result is development validation, not unbiased test performance or proof of the unique full-epoch collapse mechanism.

## Saving and resuming

The default per-session budget is 90 minutes, with a two-minute margin for saving. The worker pauses at a batch or case boundary. A single validation case may cross the remaining time budget. Input preparation and saving also consume runtime time; the budget is not a guarantee about Colab session allocation.

If training has exhausted the notebook timer, Cell 10 now explicitly starts another budget for validation using the saved checkpoints. Cells 10 and 11 share that renewed budget. This can extend the total runtime beyond the initial 90 minutes; it does not extend Colab's allocation. If validation pauses again, rerun Cell 10 followed by Cells 11 and 12 to reuse saved per-case results.

The output folder is `MyDrive/MERIT_alignment_test/runs/alignment_pair_300_v1/`. Each branch keeps one latest `progress.pt`, containing model, optimizer, scheduler, scaler, RNG, processed samples and the experiment signature. It is saved every fifty batches, after a local write and a completed copy/rename to Drive. Allow approximately 7 GB free on Drive for the two snapshots and an in-flight write, in addition to inputs. A sudden disconnect may lose updates after the last completed snapshot.

Validation stores `validation-partial.json` after each completed case. Reconnect, mount Drive, keep the same settings and `RUN_NAME`, and rerun the notebook. Completed training and saved validation cases are skipped. GPU/Torch/dependency changes or source/protocol changes are rejected to avoid mixing unmatched conditions. Start a new `RUN_NAME` only when intentionally starting a new paired experiment.

Snapshots use `merit_colab_alignment_pair_v1`, explicitly marked as partial-epoch experiments. They cannot be passed to the ordinary native-epoch trainer's resume command. `comparison.json` is marked complete only after both branches finish native validation, and checks matching initial state, sample order and non-alignment protocol settings.

Checkpoint recovery is now embedded directly in Cell 10 of [the main notebook](../notebooks/colab_alignment_pair.ipynb). Run Cells 10, 11 and 12 to check/recover both saved endpoints, validate A and B, and compare. No separate script or pasted recovery cell is required. Keep the same `RUN_NAME`, source bundle and GPU/Torch environment. If the runtime restarted, run Cells 1–7 first and then 10–12; this cannot restore local files lost when the runtime reset.

The recovery cell checks both arms' local `*-save.pt` and `*-read-*.pt` files, Drive checkpoints/pending files and prior recovered copies. It validates the full ZIP CRC, input identity, arm, batch count and saved signature before selecting a checkpoint. Original files remain unchanged; a separate content-named Drive copy is written and read back for SHA256 verification when possible. Successful verification on the mounted filesystem does not establish that cloud synchronization has finished. The cell redirects the frozen reader only in memory and then runs native validation using the saved endpoints and existing per-case results; it performs no training or source-bundle migration. If all complete copies are missing or damaged, it reports that condition without claiming recovery. The original saver logged copy completion without a read-back integrity check; its historical `SAVED` messages alone do not prove checkpoint readability.

## Rebuild and local verification

```powershell
python tools/prepare_colab_alignment_notebook.py
Copy-Item tests/test_colab_alignment_pair.py outputs/colab_alignment_pair/test_source/tests/test_colab_alignment_pair.py
Set-Location outputs/colab_alignment_pair/test_source
python -m pytest tests/test_colab_alignment_pair.py -q
```

The builder verifies the previously reviewed frozen source and embeds the new worker plus all dataset hashes. The receipt is `outputs/colab_alignment_pair/notebook-provenance.json`. Local tests cover relocation, two-device-to-one-device RNG binding, exact CPU mid-group resume, alignment-only config differences, per-case validation resume, protocol mismatch rejection, safe archive staging and notebook syntax/schema. A successful local check does not mean the notebook has run on a hosted Colab GPU.

Drive mounting follows [Colab's official I/O notebook](https://colab.research.google.com/notebooks/io.ipynb).
