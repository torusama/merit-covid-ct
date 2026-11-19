# Epoch-1 quality investigation, 2026-11-16

The epoch-1 checkpoint has a real segmentation collapse. The available controlled probes do not demonstrate a faulty fast-attention implementation, and they do not identify the exact cause of the optimization failure.

## Observed full validation

- Epoch 0: native macro Dice 0.2820627025; 1/32 empty predictions.
- Epoch 1: native macro Dice 0.0028351280; 29/32 empty predictions.
- The 32 case IDs and GT foreground voxel counts are unchanged. All saved model tensors are finite.
- These are complete native-volume results. The additional probe slices below are label-selected diagnostic slices and are not replacement validation metrics.

## Inference controls on the saved checkpoints

Kaggle audit version 2 uses the exact deployed source archive `2946fd0c9a3b06b271983866923a2ea3adb1ddc8b4fc670264875b1022ac6ac2`, with no optimizer updates. Original and factorized attention differ by at most 1.7881393432617188e-7 in foreground probability across the fixed probes.

On A-0014, slice 162, epoch-0 Dice is 0.68541067. Epoch-1 Dice is zero. Its maximum foreground probability is approximately 5.26e-19. Reverting attention, using raw logits without PCG correction, switching AMP/FP32, using batch BatchNorm statistics, or restoring epoch-0 BatchNorm statistics does not restore lesion predictions. This rules out those *inference substitutions alone* as repairs; it does not rule out their influence during earlier training.

## Backward controls on native 11 x 512 x 512 slabs

The same epoch-0 checkpoint, epoch-1 ramp, input, labels and CPU/CUDA RNG are restored for each backward pass. No optimizer steps are performed.

| Probe | Original/fast loss | Full-gradient relative L2 | Cosine |
| --- | --- | --- | --- |
| Initial empty master, A-0573 slice 57 | 0.03251933679 / 0.03251933679 | 0 | 1 |
| Positive master, A-0573 slice 17, 10,487 lesion voxels | 1.29781293869 / 1.29781293869 | 0.0008210062 (0.0821%) | 0.9999997304 |

The positive probe has nonzero MER gradients: L2 0.00584640 / 0.00584576. Their relative L2 difference is 0.00307149 (0.3071%). Gradients are finite. Twelve original-vs-factorized output/gradient unit tests also passed. These bounded probes do not prove identical complete-epoch training trajectories.

## Training conditions worth investigating, not established causes

The exact replay of epoch-1 sampling matches the saved loader generator. Training uses batch size 1 and contiguous volume blocks. Of the 127 training volumes and 8,810 masters, 62.79% of masters have empty GT; foreground occupies 0.4453% of voxels. The last 100 masters contain 81 empty masters and 57 entirely empty slabs. Such ordering and imbalance are candidate contributors, not causal proof.

BatchNorm variance ratios reach 4,402.7 and mean shifts reach 224.6 old standard deviations. Counters are exactly 8,810 -> 17,620, consistent with one update per training batch rather than double updates from recomputation. Inference-statistics controls did not rescue the positive slices, so replacing running statistics alone is not an established fix.

Epoch-average training loss is an average over changing model states and inputs; it is not the final checkpoint's training-set loss. Its reduction cannot establish improved lesion detection. The analytic uniform-prediction loss curve is a diagnostic only and does not establish that an all-background model beats the epoch-0 objective.

## Limits and next discriminating experiment

The exact cause of training collapse remains unproven. To attribute it to the fast execution change requires paired optimization from the same epoch-0 checkpoint, data order, RNG, precision and PCG schedule, measuring fixed positive/negative probes during training. No full paired epoch has been run, so neither blaming nor fully clearing the optimization change is justified. The three audit jobs changed no production weights, loss, learning rate or sampling policy.

Repository: merit-covid-ct, HEAD 13d7695. GitNexus index observed at 2026-11-16T09:06:12Z; execution conclusions above use the frozen deployed source and downloaded Kaggle artifacts directly.

## Low-LR 50-batch pilot

Kaggle `vtan2405/merit-low-lr-50-batch-pilot` completed 50/50 optimizer updates with no AMP skips. It starts from the same epoch-0 checkpoint, uses fast attention and LR 3e-5 instead of 3e-4, and keeps optimizer moments, sampling and the epoch-1 PCG schedule. Training-loop and probe time was 137.06 seconds, excluding bootstrap.

These are three label-selected positive diagnostic slices (A-0014/162, A-0319/25, A-0347/20), not full-volume validation metrics:

| Batch | Mean positive-slice Dice | Mean positive-slice recall | Empty positive predictions |
| --- | --- | --- | --- |
| 0 | 0.740259 | 0.592228 | 0/3 |
| 10 | 0.363397 | 0.249777 | 1/3 |
| 20 | 0.331518 | 0.220613 | 1/3 |
| 30 | 0.340025 | 0.228211 | 1/3 |
| 40 | 0.357183 | 0.243954 | 1/3 |
| 50 | 0.330579 | 0.219837 | 1/3 |

All three empty-slice controls retain zero false-positive voxels. All 50 training batches are from A-0573, with 36 positive masters. Thus the immediate decline in this pilot cannot simply be attributed to a majority of empty training masters. Lowering LR alone did not preserve these probe scores. The exact cause remains unresolved; case-block sampling, normalization and optimization interactions still require controlled testing. Restoring old BatchNorm statistics on the much later epoch-1 checkpoint did not fix that checkpoint, but it does not exclude an early effect of BatchNorm adaptation on this 50-batch pilot.

Latest partial checkpoint: `low-lr-pilot/checkpoints/in_progress.pt`, epoch 1, 50 batches completed. It is not an epoch-complete checkpoint. Original epoch-0 weights remain unchanged. No next job was launched during result inspection.

## BatchNorm and case-order controls, reviewed 2026-11-17

Kaggle `vtan2405/merit-bn-volume-audit` completed all five arms without numerical failures. Every arm reloaded the same epoch-0 model, independently cloned optimizer state, restored RNG, used the same fast execution and LR 3e-5, and kept the same PCG ramp. The first ten batch IDs match the completed low-LR pilot exactly. Runtime excluding preparation was 114.55 seconds.

These are three fixed label-selected positive slices with three empty controls, not cohort validation scores:

| Arm | Samples | Optimizer steps | Mean positive-slice Dice | Recall | Empty positive predictions |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 0.740259 | 0.592228 | 0/3 |
| Normal training, one volume | 10 | 10 | 0.363397 | 0.249777 | 1/3 |
| Only BN statistics update, one volume | 10 | 0 | 0.345909 | 0.233548 | 1/3 |
| Train with BN statistics frozen | 10 | 10 | 0.564016 | 0.420552 | 0/3 |
| BN-only, two volumes blocked | 20 | 0 | 0.428137 | 0.302078 | 0/3 |
| BN-only, identical pool interleaved | 20 | 0 | 0.361334 | 0.247839 | 1/3 |

The BN-only arms preserve the complete model-parameter SHA256 `e8719d7f6ba82edc89aea50b83178113ca8699be2635eb0eb2acb06d81e95850`. They change 126 BatchNorm buffers (42 layers: running mean, variance and batch count), without optimizer steps. Thus BN-statistics adaptation is a directly demonstrated cause of early diagnostic prediction degradation, independently of learned parameter changes. The one-volume controls contain seven positive masters in ten batches, so a majority of empty training slices is not needed to reproduce this early decline.

The frozen-BN training arm changes parameters but no BN buffers, and none of its three positive probes becomes empty. Its Dice still declines, however: freezing BN is a partial mitigation in this short test, not a complete repair. Interleaving exactly the same 20 samples from A-0573 and A-0199 does not improve this BN-only comparison. This does not justify claiming that all sampling strategies are ineffective.

The earlier unsuccessful replacement of BN statistics on the much later epoch-1 checkpoint remains consistent with these findings: it did not reverse the effects of an entire epoch of parameter evolution. The full collapse still cannot be attributed solely to BN from this short control.

No production checkpoint changed, no next job was launched during result inspection, and the Colab port was canceled at the user's request. The next discriminating repair trial is a bounded frozen-BN continuation from the trusted epoch-0 checkpoint with the same low LR, data order and diagnostic probes.

## Frozen-BN 50-batch continuation, reviewed 2026-11-17

Kaggle `vtan2405/merit-frozen-bn-50-batch-pilot`, version 1, completed 50/50 optimizer updates with no AMP skips. It starts from the trusted epoch-0 checkpoint, preserves optimizer moments, uses LR 3e-5 and the same epoch-1 PCG schedule and fast execution as the earlier low-LR pilot. All 50 case/master IDs match that pilot exactly. Runtime excluding preparation was 137.68 seconds. Thirteen tests passed against the exact packaged source before submission.

Downloaded source-file hashes match the local package provenance. All 126 BatchNorm buffers remained bitwise unchanged at batches 10, 20, 30, 40 and 50, including every batch counter. Training losses and gradient norms remained finite.

These are the same three fixed positive diagnostic slices, not full-volume validation scores:

| Batch | Earlier low-LR Dice | Frozen-BN Dice | Frozen-BN recall | Empty positive predictions |
| --- | --- | --- | --- | --- |
| 0 | 0.740259 | 0.740259 | 0.592228 | 0/3 |
| 10 | 0.363397 | 0.564016 | 0.420552 | 0/3 |
| 20 | 0.331518 | 0.425168 | 0.312141 | 1/3 |
| 30 | 0.340025 | 0.362037 | 0.248546 | 1/3 |
| 40 | 0.357183 | 0.243095 | 0.150783 | 1/3 |
| 50 | 0.330579 | 0.194875 | 0.118905 | 1/3 |

A-0014/162 becomes empty by batch 20. At batch 50, Dice is 0.425640 on A-0319/25 and 0.158986 on A-0347/20. All three empty-slice controls retain zero false-positive voxels throughout. Freezing BN delays the initial degradation but does not preserve these positive-slice scores; by batch 50 it performs worse than the corresponding unfrozen pilot. Degradation with fixed BN buffers demonstrates that running-statistics adaptation is not required for this longer pilot's deterioration. It does not identify the remaining cause or establish a full-cohort ranking of the two policies.

The earlier BN-only causal finding remains valid, but frozen BN is not an established repair. A full training epoch with this policy is not supported by the current diagnostic results. Further experiments should distinguish parameter-update effects under the current single-volume sampling from other causes, or directly compare original versus fast optimization with matched starting state and samples.

Artifacts are under `outputs/kaggle_frozen_bn_pilot/results_v1/frozen-bn-pilot/`: `pilot-report.json`, `comparison.json`, `source.json` and `run-status.json`. The job saved `checkpoints/in_progress.pt` remotely at epoch 1, batch 50; this is not an epoch-complete checkpoint. No production weights changed, and no further job was launched during this result review.

## Matched original-attention optimization control, reviewed 2026-11-17

Kaggle `vtan2405/merit-original-bn-50-batch-control`, version 1, completed 50/50 optimizer updates with no AMP skips. It uses original `checkpointed_mer` attention with query chunks of 128, instead of factorized attention with chunks of 4096 and no MER recomputation. The remaining training conditions match the completed frozen-BN fast pilot: original epoch-0 model, optimizer moments and RNG, LR 3e-5, frozen BatchNorm statistics, epoch-1 PCG ramp, precision settings, all 50 case/master IDs and all six probe checkpoints. The runtime backend is checked before every training batch, and its original-execution settings are recorded in the checkpoint configuration and migration metadata. Fourteen tests passed on the exact packaged source before submission.

Downloaded source hashes and the embedded fast-reference report digest match local package provenance. All 126 BN buffers remain unchanged at every ten-batch check. Losses and gradient norms are finite. Original-attention training and probe runtime was 319.35 seconds, excluding preparation, versus 137.68 seconds for the fast pilot.

These are the same three positive diagnostic slices, not full-volume validation:

| Batch | Fast-attention Dice | Original-attention Dice | Fast recall | Original recall |
| --- | --- | --- | --- | --- |
| 0 | 0.740259 | 0.740259 | 0.592228 | 0.592228 |
| 10 | 0.564016 | 0.563988 | 0.420552 | 0.420519 |
| 20 | 0.425168 | 0.421350 | 0.312141 | 0.308047 |
| 30 | 0.362037 | 0.353810 | 0.248546 | 0.240904 |
| 40 | 0.243095 | 0.232199 | 0.150783 | 0.143105 |
| 50 | 0.194875 | 0.186467 | 0.118905 | 0.113668 |

Both implementations make A-0014/162 empty by batch 20, while all three empty-slice controls retain zero false-positive voxels. Their initial probe metrics match exactly, and their first two training losses match exactly. Later trajectories differ numerically; the largest mean-Dice gap among measured checkpoints is 0.010897 at batch 40. At batch 50, original attention is 0.008408 below fast attention. Raw-logit probe Dice also deteriorates, so the PCG correction applied during evaluation alone does not explain the observed pattern.

This bounded matched comparison reproduces the diagnostic degradation with original attention and fixed BN statistics. Reverting to original attention is therefore not a demonstrated repair, and fast attention is not necessary for this 50-batch failure pattern. This does not prove full-epoch optimization equivalence, clear every fast-execution change under other settings, or establish cohort-level performance. Both runs still consume their first 50 batches from one training volume, A-0573.

The next sampling control can use an identical pool of training samples drawn from multiple volumes, with independent epoch-0 restarts for blocked versus interleaved ordering and otherwise matched settings. Such a comparison would test ordering without changing the sample pool. It remains a proposed experiment, not an established fix. No further job was launched during this review.

Downloaded JSON reports and logs are under `outputs/kaggle_original_bn_pilot/results_v1/`; `original-bn-pilot/comparison.json` and `train-comparison.json` contain matched probe and per-batch training comparisons. The partial checkpoint remains on Kaggle. No production checkpoint changed.

## Matched multi-volume optimization order, reviewed 2026-11-17

Kaggle `vtan2405/merit-multivolume-order-control`, version 1, completed both 50-update arms (100 optimizer updates total), with no skipped AMP steps or numerical failures. The two arms use exactly the same 50 unique masters from five training volumes, selected without label-based filtering: ten masters per volume from the saved loader's case and master permutations. Both use fast attention, LR 3e-5, frozen BatchNorm statistics and the same epoch-1 PCG ramp.

The selected volumes are A-0573, A-0199, A-0377, A-0255 and A-0270. Their positive-master counts are 7, 3, 5, 5 and 4 respectively: 24/50 positive masters in total. The blocked arm consumes all ten samples of each volume before moving to the next; the interleaved arm cycles through the five volumes, keeping each volume's internal order unchanged.

Initial model, optimizer, scheduler, scaler, CPU/CUDA RNG and loader-generator hashes match across arms. Initial probes match exactly. All 126 BN buffers remain unchanged in both arms. Cached input tensors and the shared restart state remain unchanged after each arm. Source-file hashes and the embedded reference-report digest match local package provenance. The blocked arm's first ten training-loss dictionaries and batch-10 probe results exactly reproduce the earlier frozen-BN pilot. Fifteen tests passed on the exact packaged source before submission.

These are the same three positive diagnostic slices, not full-volume validation:

| Batch | Blocked Dice | Interleaved Dice | Blocked recall | Interleaved recall |
| --- | --- | --- | --- | --- |
| 0 | 0.740259 | 0.740259 | 0.592228 | 0.592228 |
| 10 | 0.564016 | 0.811092 | 0.420552 | 0.691934 |
| 20 | 0.487790 | 0.857380 | 0.364783 | 0.806032 |
| 30 | 0.485660 | 0.797335 | 0.362944 | 0.669638 |
| 40 | 0.563759 | 0.740196 | 0.425624 | 0.592156 |
| 50 | 0.761274 | 0.741026 | 0.623454 | 0.593018 |

No positive probe becomes empty at any measured checkpoint in either arm. The three empty-slice controls retain zero false-positive voxels throughout. Runtime excluding initial pool preparation was 118.20 seconds for blocked updates/probes and 122.16 seconds for interleaved updates/probes; total reported experiment time, including state checks, was 299.16 seconds.

Interleaving gives substantially better early diagnostic scores in this pool, while the blocked arm recovers and ends 0.020248 Dice higher at batch 50. Thus the experiment demonstrates order-sensitive trajectories, not a consistent final advantage for interleaving. At intermediate checkpoints the two arms have consumed different prefixes of the same final pool, as intended by this ordering comparison. Both arms finish near or above their initial positive-slice mean Dice, unlike the earlier 50 consecutive updates from A-0573 alone. That historical comparison changes the sample pool and lesion composition as well as volume diversity, so it cannot establish diversity alone as the cause of recovery.

This is an encouraging bounded result, not an established fix for the full epoch-1 collapse. BN remains frozen throughout, only one five-volume pool and one seed were tested, and no complete validation volume was evaluated. A longer, predeclared multi-volume pilot from the trusted epoch-0 checkpoint, with full native-volume validation at its endpoint, is the next repair check. The batch-20 peak on three selected slices is not a cohort score or a basis for selecting a production checkpoint.

Artifacts are under `outputs/kaggle_order_optimization_pilot/results_v1/order-optimization-pilot/`: `order-report.json`, `comparison.json`, `source.json` and `run-status.json`. This diagnostic job saves no model checkpoints. No production weights changed, and no further job was launched during result review.

## 300-update repair pilot with native validation

Kaggle `vtan2405/merit-multivolume-300-native-validation`, version 1, was submitted and completed on 2026-11-17. Results are reviewed below. Eleven tests passed on the exact packaged source, including snapshot integrity, unchanged BN buffers, optimizer update counts and agreement between casewise and full-cohort native evaluation.

The predeclared budget is 300 unique masters from 30 seeded training volumes, ten masters per volume without label filtering. Samples are interleaved within six consecutive groups of five volumes; the first 50 identities match the reviewed interleaved arm exactly. The job restarts from the trusted epoch-0 checkpoint with optimizer moments preserved, fast attention, frozen BN, LR 3e-5 and the epoch-1 PCG ramp. It measures the fixed slice probes and saves a pilot snapshot every 50 updates. The endpoint snapshot is reloaded before evaluating all 32 validation cases on their complete native grids, with the same fixed threshold as the historical epoch-0 baseline.

The checkpoint uses explicit `merit_multivolume_pilot_v1` metadata binding the custom sampling plan and processed batch count. It is not an epoch-complete checkpoint or a checkpoint for the native full-epoch sampler. Validation reports compare macro Dice, empty predictions and per-case scores with the original epoch-0 history, requiring identical case IDs and GT foreground voxel counts. The T4 job has a maximum runtime of 10,800 seconds. Package and submission receipts are under `outputs/kaggle_multivolume_300/`.

### Completed 300-update results, reviewed 2026-11-17

The job completed all 300 optimizer updates without AMP skips or nonfinite losses/gradient norms. The executed plan contains 300 unique masters from 30 volumes, exactly ten per volume; 118 masters have positive GT. The first 50 identities match the earlier interleaved arm, and its batch-50 diagnostic Dice is reproduced exactly. All six BN checks report zero buffer changes. Source hashes, the epoch-0 parent digest and the embedded order-reference report digest match local provenance. The shared restart state remains unchanged.

Complete native validation covered all 32 expected cases, with identical IDs and per-case GT foreground voxel counts to epoch 0. Evaluation used the saved endpoint snapshot, the unchanged native evaluator and threshold 0.5.

| Native-volume result | Epoch 0 | Multi-volume pilot, 300 updates |
| --- | --- | --- |
| Group/volume macro Dice | 0.282063 | 0.303006 |
| Macro IoU | 0.193720 | 0.205926 |
| Macro precision | 0.502044 | 0.330791 |
| Macro recall | 0.365391 | 0.537448 |
| Empty predictions | 1/32 | 0/32 |
| Zero-Dice cases | 3/32 | 1/32 |
| Total false-positive voxels | 3,142,381 | 4,860,016 |
| Voxel-pooled Dice | 0.252257 | 0.249876 |

Macro Dice improves by 0.020943 (2.094 percentage points), with 17 cases improving, 14 declining and one unchanged. Recall improves, but precision declines and total false-positive voxels increase by 54.66%. Voxel-pooled Dice declines slightly. This is a mixed precision/recall tradeoff, not an across-the-board improvement. The largest Dice regression is A-0694, 0.678655 -> 0.326858; the largest improvement is A-0666, 0.387637 -> 0.697640.

The three selected positive-slice mean Dice values are 0.740259, 0.741026, 0.818451, 0.837625, 0.662085, 0.844334 and 0.862759 at batches 0, 50, 100, 150, 200, 250 and 300. None of these positive probes becomes empty; the three selected empty controls still have zero false-positive voxels at every probe. Their strong endpoint score and empty-control results do not represent the complete-volume false-positive behavior shown above.

The all-background failure is not reproduced at this bounded endpoint under the combined frozen-BN, low-LR, multi-volume protocol. No single change can be credited as the sole cause, and the result does not establish a repair for a complete training epoch or a robust improvement across seeds. Before increasing the training budget, the next useful analysis is to inspect where the additional false-positive volume occurs, especially in cases whose full-volume Dice regressed. A checkpoint should not be selected from the three-slice peaks.

The reported training/checkpoint phase takes approximately 592.34 seconds; complete native validation takes 2,510.55 seconds (41.84 minutes). Total experiment time excluding bootstrap is 3,102.89 seconds (51.71 minutes). The endpoint pilot snapshot is retained on Kaggle as `multivolume-300-pilot/checkpoints/pilot-progress.pt`; it is a partial epoch-1 pilot with its custom sampling plan, not an epoch-complete or native-sampler resume checkpoint.

Downloaded JSON/log artifacts are under `outputs/kaggle_multivolume_300/results_v1/`. `result-review.json` summarizes the checks and metrics, and `case-comparison.csv` records all 32 per-case Dice, precision, recall and voxel-count comparisons. No production checkpoint changed, and no further job was launched during this review.

## Read-only FP confidence and spatial audit

Kaggle `vtan2405/merit-pilot300-fp-confidence-audit`, version 1, was submitted and completed on 2026-11-17. Results are reviewed below. Ten tests passed against the exact packaged source, including native threshold-0.5 replay, fixed confidence-bin counts, connected-component FP accounting, orientation transforms, unchanged model/checkpoint state and montage/probability export.

The job reads the saved 300-update snapshot and audits the same 32 native validation cases. It scans predeclared thresholds 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 and 0.99; these are validation diagnostics, not a new production threshold or an unbiased test score. It computes probability histograms separately for GT foreground/background and describes 26-connected prediction components with and without GT overlap. No component filtering is applied. Replay Dice at threshold 0.5 is checked against the preceding full validation report, whose digest is pinned.

Eight visual cases are selected before running: A-0014, A-0112, A-0133, A-0397, A-0585, A-0643, A-0666 and A-0694. They comprise the five largest FP increases, two largest Dice regressions and one improvement control from the already reviewed reports. Montages show CT, GT, threshold-0.5 prediction, TP/FP/FN and probability at slices with maximum FP, FN and GT area. Compressed probability arrays with original/canonical geometry are retained for these cases.

The audit performs no optimization, saves no model checkpoint, and verifies unchanged parameters and buffers after inference. Its GPU time limit is 10,800 seconds. Local package and submission receipts are under `outputs/kaggle_fp_confidence_audit/`.

### Completed confidence/spatial audit, reviewed 2026-11-17

All 32 native validation cases completed. Source hashes and the pinned prior pilot/baseline report digests match local provenance. Model state hashes before and after inference are identical (`cb5bbc4dead5bff06a8b16a19739afc6df00d46bea049f0b6704f9f5d9e98323`); optimizer updates are zero. At threshold 0.5, each case's Dice and TP/FP/FN exactly reproduce the preceding pilot validation. Histogram foreground/background counts and component FP accounting agree with the native metrics. Eight montages were downloaded. Audit runtime was 2,711.97 seconds (45.20 minutes).

| Diagnostic threshold | Native macro Dice | Macro precision | Macro recall | FP voxels | FN voxels |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 0.303006 | 0.330791 | 0.537448 | 4,860,016 | 713,979 |
| 0.60 | 0.305055 | 0.338109 | 0.528187 | 4,604,976 | 731,607 |
| 0.70 | 0.307244 | 0.347054 | 0.515772 | 4,291,784 | 754,595 |
| 0.80 | 0.310058 | 0.359591 | 0.496941 | 3,837,706 | 790,168 |
| 0.90 | 0.315219 | 0.385168 | 0.455879 | 2,925,204 | 867,924 |
| 0.95 | 0.317571 | 0.415588 | 0.399791 | 1,970,977 | 970,250 |
| 0.99 | 0.289519 | 0.492339 | 0.262280 | 656,946 | 1,223,195 |

No volume has an entirely empty prediction at any scanned threshold. The largest scanned macro Dice occurs at 0.95, but this is a validation-selected operating point, not a new production threshold or unbiased performance estimate. Its gain over 0.5 is only 0.014565, with recall reduced by 0.137657. Sixteen cases improve and fifteen decline in Dice at 0.95 relative to 0.5, with one unchanged. Raising the threshold does not consistently repair individual cases.

At threshold 0.5, 1,212,573 FP voxels (24.95%) belong to 26-connected prediction components with no GT intersection. The other 3,647,443 FP voxels (75.05%) are in prediction components that do intersect GT. These are descriptions using GT, not deployable component-selection rules. They show that discarding disjoint speckles alone cannot account for all FP volume. Some disjoint components are themselves large: A-0112 has one of 429,453 voxels, and A-0643 has disjoint components of 224,608, 97,621 and 93,138 voxels. A simple small-component filter is therefore not an established repair.

Of the FP voxels at threshold 0.5, 40.55% remain at threshold 0.95 and 13.52% at 0.99. Thus the mistakes are not confined to probabilities just above 0.5. Inspection of exported diagnostic montages shows broad predictions outside the annotated regions: A-0585 predicts much of both visible lung fields on slices 21/26 despite much smaller GT regions; A-0643 includes broad off-body/background predictions at slices 32/70. These selected images establish errors relative to supplied GT, not annotation completeness or a medical diagnosis. A-0014 and A-0694 likewise show false-positive regions alongside true-positive regions. The improvement control A-0666 retains non-GT predictions even though its whole-volume Dice improved.

The audit supports investigating how training distinguishes foreground from difficult background, rather than treating a higher threshold or speckle removal as a complete fix. A future bounded paired training experiment could compare the current objective with a predeclared background penalty or training-only hard-negative selection, starting from identical states and using the same samples, with endpoint full-native validation and fixed threshold 0.5. Such changes are hypotheses, not demonstrated repairs. The present audit does not identify the loss, labels, preprocessing or architecture as the sole root cause.

Downloaded reports, `threshold-curves.png` and eight montages are under `outputs/kaggle_fp_confidence_audit/results_v1/fp-confidence-audit/`. Compressed probability volumes remain in the Kaggle output. `result-review.json` and `threshold-comparison.csv` summarize the verified results locally. Production weights and threshold remain unchanged, and no further job was launched during this review.

## Submitted comprehensive cause-and-repair suite

At the user's request to consolidate the remaining investigations, Kaggle `vtan2405/merit-comprehensive-cause-and-repair`, version 1, was submitted on 2026-11-17. Results are pending. Twenty-five tests passed in 70.16 seconds against the exact packaged source, covering every screening and confirmation arm, native geometry checks, component-loss gradients, typed endpoint snapshots, an isolated worker subprocess and failure/repair-gate accounting. The job uses a T4 with a 39,600-second maximum runtime; this is a multi-hour suite, not a new completed training epoch.

The job verifies the SHA256 identities of the original completed epoch-0 checkpoint and the saved 300-update snapshot. Nine reviewed source/helper files remain unchanged. Historical verified evidence is embedded with report digests. Each optimization arm starts independently from epoch 0; arms execute in isolated processes so one failed allocation or numerical failure does not invalidate other independent tasks. Data-verification failure blocks dependent training. Reports and worker logs persist after each task, and failed/missing tasks remain explicitly inconclusive.

The data audit reads actual train/validation CT and binary masks, checks finite intensities and exact original/canonical GT round trips, and records per-case intensity quantiles, border intensities and foreground/slice counts. Test-partition content is not analyzed or evaluated beyond the preexisting source-byte identity checks. Gradient audits at epoch 0 and the saved 300-update checkpoint use training-only positive, empty and mined hard-negative samples. They measure weighted loss terms, per-module gradient norms and output-head gradient directions without optimizer updates. Hard negatives are selected from a fixed training-only pool, never validation.

### Predeclared 50-update screening arms

Thirteen arms use the same fixed diagnostic cases and matching training pools where the intervention permits:

1. Fast execution, frozen BN, mixed volumes, LR 3e-5 (reference).
2. Original checkpointed attention, otherwise matched.
3. Updating BN statistics, otherwise matched to reference.
4. Blocked multi-volume order on the same pool.
5. Presence/alignment/salience loss weights all zero.
6. TEP bypassed.
7. MER bypassed.
8. PCG bypassed.
9. All three MERIT modules bypassed.
10. Additional background penalty, weight 0.2: mean softplus(foreground-minus-background logit) on valid GT-background slab voxels.
11. Adam optimizer moments reset, otherwise matched.
12. Legacy fast execution with updating BN, original LR 3e-4 and 50 consecutive samples from the first native volume.
13. Original attention under the same legacy conditions.

The diagnostic probes expand to twelve predeclared validation cases, including the prior three controls, FP/Dice-regression cases, an improvement control and two further cases. They remain selected-slice diagnostics, not cohort scores.

### Predeclared 300-update confirmations

Six arms are selected before screening results: reference fast, original attention, auxiliary losses off, background penalty, optimizer reset and all MERIT modules bypassed. Each restarts at epoch 0, executes 300 updates, saves a typed endpoint model and performs full native validation on all 32 cases at threshold 0.5. Confirmation arms are not chosen by the best short-run probe score. The mixed plan retains thirty volumes, ten samples each, with round-robin ordering within consecutive groups of five. The maximum optimizer-update budget is 2,450 across all screening and confirmation tasks.

A candidate passes the predeclared repair checks only if native macro Dice does not decline versus the suite reference, macro precision improves by at least 0.05, total FP voxels fall by at least 20%, macro recall drops by no more than 0.05, and empty predictions do not increase. These are development-data checks, not claims of unbiased test performance or production readiness. Per-case validation and complete arm specifications remain available regardless of the gate result.

Module bypasses retain weights trained with MERIT and remove unavailable module auxiliary outputs/losses. They are bounded interventions on a trained checkpoint, not a backbone trained from scratch, and cannot alone establish that the entire proposed architectural improvement is harmful. The original-versus-fast confirmation tests execution under matched optimization, not every possible training trajectory. One seed and partial-epoch budgets cannot settle full-epoch causality or robustness. Data consistency checks do not establish clinical annotation completeness.

The job emits `suite-verdict.json`, `suite-verdict.md`, `experiment-design.json`, `task-failures.json`, per-arm model/report/validation artifacts and worker logs. Package and submission receipts are under `outputs/kaggle_comprehensive_diagnosis/`. Production weights, source images, splits and threshold are unchanged.

### User-canceled suite: recovered partial results, reviewed 2026-11-17

The user stopped the job after exhausting available Kaggle GPU time. Kaggle reports `CANCEL_ACKNOWLEDGED`. The exported logs reach approximately 23,364.74 seconds (6.49 hours) after notebook start. The workload was too expensive for the user's remaining quota; future GPU work must account for available time and reuse completed native predictions rather than repeat completed validation.

Downloaded artifacts preserve all thirteen completed 50-update screens, the data/gradient audit, four complete 300-update confirmations and a partially validated fifth confirmation. `task-failures.json` is empty: the saved tasks show no technical failure. The suite itself remains **partial**, not complete. Its root final `run-status.json` was never written because cancellation interrupted the optimizer-reset confirmation. All-module-bypass confirmation had not started.

Source-file hashes, the original epoch-0 parent digest and the saved pilot-300 digest match package provenance. Every completed confirmation has 300 finite, unskipped updates, matching sample identities and identical initial model/optimizer/scheduler/scaler/RNG/loader state. Frozen BN buffers remain unchanged. Every complete native result contains the original 32 case IDs and identical GT foreground counts at threshold 0.5. The fast-reference metrics exactly reproduce the previous 300-update run.

| Completed full-native confirmation | Macro Dice | Macro precision | Macro recall | FP voxels | Empty predictions |
| --- | --- | --- | --- | --- | --- |
| Reference fast | 0.303006 | 0.330791 | 0.537448 | 4,860,016 | 0/32 |
| Original attention | 0.305698 | 0.348248 | 0.503380 | 4,217,664 | 0/32 |
| Auxiliary losses off | 0.311421 | 0.338413 | 0.529153 | 4,555,926 | 0/32 |
| Background penalty, weight 0.2 | 0.305773 | 0.356731 | 0.514245 | 4,575,707 | 1/32 |

Original attention changes macro Dice by only +0.002692 relative to fast attention under these matched 300 updates. It reduces FP by 13.22% with reduced recall, but does not remove the previously observed foreground/background problems. This is evidence against attributing the principal failure pattern uniquely to fast attention; it is not proof of identical complete-epoch trajectories. Both legacy-condition screens also deteriorate: selected-positive-slice mean Dice starts at 0.511427 and ends at 0.244849 with fast execution and 0.290381 with original attention after fifty consecutive single-volume updates at LR 3e-4 and updating BN.

Disabling auxiliary losses gives the largest complete native Dice among the tested confirmations, but only +0.008415 over the fast control, with FP down 6.26%. The background penalty reduces FP by only 5.85% and introduces one empty prediction. **None of the completed candidates meets every predeclared repair gate.** These small differences do not establish a production repair or justify treating auxiliary loss as the sole cause.

The optimizer-reset arm completed all 300 training updates, saved its endpoint model and evaluated 27 of 32 validation cases before cancellation. The remaining cases are A-0643, A-0659, A-0666, A-0694 and A-0696. On exactly the same 27-case subset, reset-optimizer macro Dice is 0.297464 versus 0.298617 for the fast reference; FP is 3,860,556 versus 3,822,588. This partial subset supplies no improvement signal, but cannot substitute for the missing full-cohort result. If its saved model remains retrievable, completed training can be reused to finish only those five predictions. The all-module-bypass confirmation has no 300-update or full-native result.

### Screening, data and gradient observations

On the twelve predeclared selected-positive-slice probes, final screening mean Dice is 0.526153 for the fast reference, 0.531893 for original attention, 0.406994 with live BN, 0.536347 with blocked multi-volume order, 0.481930 with TEP bypassed, 0.528614 with MER bypassed, 0.496168 with PCG bypassed and 0.455081 with all MERIT modules bypassed. Module-bypass starting metrics can differ; these endpoint slice scores alone cannot rank the complete architecture. In particular, the missing all-module native confirmation and lack of a from-scratch backbone baseline prevent attributing the entire proposed architectural improvement as the cause of failure.

The data audit completes 159 train/validation volumes: 127 training and 32 validation. All recorded GT original/canonical geometry round trips are exact; CT arrays are finite and labels binary. Training has 8,810 masters, 3,278 positive masters and foreground voxel fraction 0.00445276; validation foreground fraction is 0.00269816. Native scaled CT ranges vary substantially, including [-9,653, 27,503] on training A-0120 and a maximum of 32,737 on training A-0295. This warrants examining native intensity handling but does not establish an encoding bug or justify applying an assumed HU window. The checks verify structural consistency, not clinical completeness or semantic correctness of annotations.

All 72 gradient measurements are finite: six terms on six training-only samples at each of epoch 0 and pilot 300. Output-head alignment-loss gradients oppose segmentation gradients on some samples; for empty-master A-0011/208 at epoch 0, the cosine is approximately -0.8675 and weighted alignment full-gradient L2 is 0.483 versus segmentation L2 0.0254. Auxiliary terms can dominate gradients in selected regions even when their scalar weighted loss is smaller. These bounded measurements identify an optimization interaction worth inspecting, not a unique mechanism for all predictions; an empty master can still have positive context labels. The auxiliary-loss-off confirmation's modest gain limits any claim that this interaction alone explains the failure.

The established findings remain: BN adaptation can independently damage the early diagnostic scores; data/order/optimization conditions matter; substantial high-confidence false-positive regions persist; fast attention alone is not a demonstrated explanation; and none of the completed repairs is sufficient. There is no demonstrated NaN/Inf or GT-orientation implementation failure in the audited paths. The stopped job cannot resolve all architectural causality or full-epoch behavior.

Recovered reports and logs are under `outputs/kaggle_comprehensive_diagnosis/results_stopped_v1/`. `stopped-result-review.json` records the verified completed and incomplete coverage; `completed-confirmations.csv` contains the four complete native comparisons. `recovery-plan.json` lists the five remaining reset-optimizer validation cases and the unstarted all-module confirmation. No further GPU job was launched and no production weights or threshold changed during this review.

After the JSON/log download completed, subsequent attempts to retrieve endpoint model weights and recheck status were rejected by Kaggle with `Permission 'kernels.get' was denied`, both for the kernel slug and explicit version 1. This does not invalidate already downloaded artifacts, but current remote checkpoint retrieval is unverified. No endpoint-model weights were downloaded during this review; recovery of the five missing predictions is conditional on regaining access to the saved model.

### CPU-only conditional alignment-gradient check

A subsequent offline check uses the exact loss source shared by the working tree and frozen deployment (SHA256 `eb9aa75ca8d6c22349b9bbb2e473de3112d850bc5b0e342264aa7ac7097d6b97`). It constructs valid MERIT outputs on an 11 x 512 x 512 grid with all-background GT, raw peak foreground probability 0.2, the epoch-1 PCG ramp 1/30 and an incorrectly high predicted presence of 0.8. It performs no model training or optimizer updates.

At the master-plane peak, segmentation contributes a positive foreground-logit gradient of approximately 2.50663e-7 (gradient descent lowers the foreground logit), while weighted alignment contributes -8.72727e-4. The total is -8.72477e-4, so the output-level descent direction raises foreground on a GT-background voxel. Setting only alignment weight to zero reverses this total gradient to the desired direction. With a correctly low presence of 0.05, the original alignment also lowers foreground. This demonstrates a conditional objective conflict caused by coupling max raw foreground probability to an erroneous learned presence estimate; it is not proof that every network parameter update follows that direction or that this alone causes the full epoch collapse.

This gives a concrete offline target before further GPU work: inspect how alignment should supervise empty-GT slices and verify that a proposed alternative cannot raise foreground solely to match erroneous presence. A targeted alignment correction or alignment-only ablation would still require controlled native validation before being called a repair. The existing all-auxiliary-loss-off result improved Dice only modestly and remains relevant counterevidence against treating alignment as the unique cause.

The same offline review finds that extreme CT minima/maxima occupy the tails: the training 99th-percentile CT values range from 232 to 688, while individual maxima can exceed 30,000. The extreme maxima alone do not establish globally incompatible intensity scales or justify changing preprocessing for a saved checkpoint.

The reproducible CPU diagnostic is `outputs/kaggle_comprehensive_diagnosis/check_alignment_gradient.py`; measurements are saved in `alignment_gradient_counterexample.json`. No production source, checkpoint or threshold was modified, and no new Kaggle job was launched.

### Colab notebook for the targeted alignment-only comparison

At the user's request to move testing from Kaggle to Colab, a self-contained notebook was created at `notebooks/colab_alignment_pair.ipynb`, with instructions in `docs/colab_alignment_pair.md`. It has seventeen cells and embeds the reviewed source archive, helper overlays, the Colab worker and all 199 CT/mask decompressed hashes. It requires dataset and the original epoch-0 checkpoint/history on the user's Google Drive, rather than Kaggle authentication or GPU quota. The notebook has not been executed in the user's hosted Colab account.

Only two arms are run: original alignment weight 0.05 and alignment weight zero. Presence and salience losses remain unchanged. Both use the same epoch-0 state, fast execution, frozen BN, LR 3e-5, epoch-1 PCG ramp and 300-master multi-volume plan. Endpoint validation covers all 32 native volumes at fixed threshold 0.5. This isolates alignment, unlike the preceding all-auxiliary-loss-off arm, but is still a development-data ablation and not an established repair.

An explicit content-verified portability migration changes data paths and associated file/manifest metadata while retaining all case/group/split identities and immutable model/preprocess/loss/training settings. The original two-GPU CUDA RNG list is reduced to stream zero when Colab has a single GPU. Cross-runtime exact historical replay is not claimed; the two Colab arms must share an environment.

Experiment-specific snapshots retain model, optimizer, scheduler, scaler, RNG, sampling plan and a signature binding source, input hashes and protocol. They are saved locally then copied into the Drive output with a pending-file/rename pattern, keeping one current snapshot per arm. Native validation persists each completed case and resumes only missing cases. GPU/Torch/dependency or source/protocol mismatches are rejected. Snapshots are typed `merit_colab_alignment_pair_v1`, not completed epochs or ordinary native-trainer resume files.

The notebook defaults to a ninety-minute per-session budget with a two-minute saving margin; work pauses at batch/case boundaries. Hosted runtime availability and time are not guaranteed. Completed training and validation can be reused on reconnect under the same experiment settings and environment. The builder is `tools/prepare_colab_alignment_notebook.py`, and the worker is `tools/colab_alignment_pair.py`. Bundle receipts and the frozen local verification tree are under `outputs/colab_alignment_pair/`. Nine tests passed in 40.60 seconds on the final frozen notebook source, including exact CPU mid-group resume and per-case validation reuse; notebook schema, Python cells and embedded source hashes were also verified. No Colab/GPU job was launched during notebook creation.
