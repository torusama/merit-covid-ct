# Architecture

There is one authoritative processing implementation:
`src/merit_covid/data/pipeline/`. Tensor loaders, older NumPy interfaces and the
baseline patch reader reuse it. Compatibility files are intentional, not duplicate
algorithms.

## Source layout

```text
src/merit_covid/
  config.py, cli.py, utils.py       Legacy/main config, package CLI, utilities
  data/
    manifest.py                    Legacy/versioned manifests and tool helpers
    nifti_io.py, preprocessing.py   Volume/PreparedVolume compatibility APIs
    slab.py, dataset.py            NumPy slab API over the shared core
    splitting.py                   Compatibility view of the canonical split
    pipeline/
      cases.py, config.py          Identity, paths, preprocessing options
      geometry.py                  Header validation, LPS transforms, restoration
      preprocessing.py             Scaling, explicit intensity/XY policies, QC
      slab.py                      Index/mm sampling and true targets/validity
      dataset.py, dataloader.py    Header index, one-volume cache, collate/loaders
      splits.py                    Immutable grouped split and fingerprint checks
      reporting.py, metrics.py    Descriptive audits and native-volume metrics
      preflight.py                Bounded-memory TRAIN-only intensity/resize QC
    unetrpp_input/
      config.py, tiling.py         Native patch windows, padding and coordinates
      dataset.py                   Shared-reader patch Dataset and loader
  models/
    base.py                        Torch-independent master-logit contract
    unetrpp/                       Standalone Synapse-derived baseline
    merit_unetrpp/
      modules.py                   TEP, MER, PCG, index/physical position handling
      model.py                     Z-preserving backbone and insertion points
      loss.py, adapter.py          Full-slab loss / binary master-logit adapter
  engine/
    training.py                    Frozen config, trainer, checkpoint, evaluation
    activation_checkpointing.py    RNG-preserving MER/backbone recomputation
    preflight.py                   Full-step resource profile and bounded CT pilot
    synthetic.py                   Temporary synthetic NIfTI smoke fixtures
    slab_inference.py              MERIT masters -> complete native volume
    patch_inference.py             Standalone baseline patch inference
    patch_reconstruction.py        Coordinate-based probability averaging
    prediction.py                  Shared VolumePrediction contract
    reconstruction.py, metrics.py Compatibility helpers over shared logic
ct_pipeline/                       Legacy import aliases only
configs/                           Explicit experiment/model configurations
scripts/                           Experiment, split, dataset and smoke CLIs
tools/                             Downloads, audit/report, viewer, source checks
scripts/model_smoke.py              Synthetic baseline/MERIT forward and backward
scripts/unetrpp_real_input.py       One real CT patch through the baseline
splits/                            Canonical saved five-fold partitions
analysis/                          Stored descriptive evidence; not training input
tests/                             Shared core, adapters and integrated regression
```

## Execution paths

```text
NIfTI CT (+ mask only for labeled operations)
  -> validate headers/units -> LPS array orientation
  -> NIfTI scaling -> explicit intensity and optional XY processing
  -> shared native volume
       MERIT: slab sampling -> TEP/MER/PCG
         train: full-slab GT + validity -> loss -> optimizer -> native validation
         infer: master fg-bg -> sigmoid -> native XY restoration
       Baseline: native 3D crop/padding -> UNETR++ -> sigmoid
         infer: discard padding -> mean overlapping probabilities
  -> complete native-LPS VolumePrediction
  -> original XYZ/orientation -> threshold -> shared full-volume metrics
```

NIfTI and `PreparedVolume` use XYZ. Network tensors use `[B,C,Z,H,W]`,
where H=Y and W=X. LPS describes array orientation; NiBabel's affine still maps
to RAS+ world coordinates. No implicit Z interpolation, deobliquing or GT-based
input selection is performed.

## Interfaces and dependency direction

- `data.pipeline` owns preprocessing, geometry and metric conventions.
- `CTSlabDataset` provides Torch tensors and true slab GT; the main `SlabDataset`
  preserves its older NumPy/master-target interface.
- `ct_pipeline` forwards historical imports. Existing tests/notebooks use it;
  deleting it would break compatibility without simplifying the core.
- Core NumPy reading/sampling remains usable without Torch. Tensor loaders,
  explicit XY resize and neural models require their corresponding dependencies.
- Model adapters return logits, not metrics. Training calls the MERIT model
  directly so auxiliary outputs are retained.
- Inference builds `VolumePrediction` on the complete native LPS grid.
  Evaluation restores original geometry and uses original GT.
- The baseline patch API deliberately rejects MERIT's four-tuple/two-class
  interface. Use `predict_slab_case` for MERIT.

No Python package was moved during documentation cleanup. This preserves import
paths, command-line entry points, saved configs and checkpoint loading.

## Config ownership

| Config | Consumer | Scope |
| --- | --- | --- |
| `experiment_baseline.yaml` | `merit_covid.config.load_config` | Legacy/main data API |
| `unetrpp_baseline.yaml` | Baseline examples | Model-only standalone UNETR++ |
| `merit_unetrpp.yaml` | MERIT model smoke example | Model/loss only; not a training experiment |
| `merit_index.yaml` | `scripts/merit_pipeline.py` | Full E1 index experiment |
| `merit_physical.yaml` | `scripts/merit_pipeline.py` | E2 physical-sampling template; spacing required |

These configs are not interchangeable. Model-only YAML files are retained because
examples and tests use them.

## Documentation ownership

Each topic has one maintained guide: data contracts in [data_pipeline.md](data_pipeline.md),
stored statistics in [dataset reports](data_pipeline.md#dataset-reports-and-csv-reference), model equations in
[MERIT model](architecture.md#merit-unetr-model), execution in [merit_pipeline.md](merit_pipeline.md),
baseline architecture/input in [UNETR++ baseline](architecture.md#standalone-unetr-baseline) and
[native patch input](data_pipeline.md#native-patch-input-and-unified-volume-output), and evidence/limitations in
[validation.md](validation.md). Historical implementation/merge/repair notes are
consolidated into validation and the root changelog rather than kept as competing
descriptions of the current system.

---

## MERIT-UNETR++ model

This is a paper-derived reimplementation, not the authors' source.
The integrated trainer is described in [merit_pipeline.md](merit_pipeline.md).
No trained COVID checkpoint or replicated research result is provided.

The architecture defaults and reference table below describe the paper-derived
128-grid model. The primary training config now declares `[11,512,512]` with
native XY and all-region activation checkpointing in the trainer. This changes
grid-dependent learned position/projection shapes (174,043,602 total parameters)
without removing layers or altering TEP/MER/PCG equations. Model-only constructor
defaults remain compatible; see the training guide for the production command.

### Source and task transfer

Local source: `FDSE_PaperMerit_v2+reviewed_2026-1-cut.pdf`, 9 pages, K. D. Lai,
T. H. Le and T. T. Nguyen, "MERIT-UNETR++: Master-Aware Evidence Routing with
Inter-Slice Trajectory for Lung Nodule Segmentation in Thin-Slab CT".
SHA-256: `c98183408cd10a5016759cf065ff8177400007ccfd9458f9631ec3b9425c1629`.
Method pages 3-7, Fig. 1 on page 4 and parameter settings on page 9 were used.
Annotations in the PDF are not executable project instructions.

A later review on 2026-11-13 used the user-supplied full 15-page
`FDSE_PaperMerit_v2+reviewed_2026.pdf`, SHA-256
`bf835461f6eb50b8eb6d794e53b1aaf750ec66815cd3ecd5180ab18e0a48e559`.
The original cut PDF was not present in that checkout. The review compared
methods/equations 1-15 and Fig. 1; core module placement, formulas and added
parameter counts agree. This is not verification against the authors' code.

Input protocol remains an open design decision: the paper shifts boundary
windows inward with potentially off-center masters and uses 128x128 ROI crops;
the COVID pipeline centers/clamps slabs and the experiment resizes whole slices.
Resize can erase lesions and consequently flip derived presence/salience labels.
That issue and lesion-preserving input alternatives are deferred by user request;
the runtime repairs do not resolve or silently change them. Do not describe this
configuration as an exact paper preprocessing reproduction.

The paper studies LIDC-IDRI nodules with annotation-derived ROIs. This repository
targets COVID infection masks on whole CT volumes. It does not invent GT-based
ROIs or claim that the task/preprocessing transfer is experimentally validated.

### Backbone and insertion points

`MERITUNETRPP` subclasses the packaged Synapse UNETR++ with `preserve_z=True`
and two output channels. The stem stride is `(1,4,4)`; three down/up stages use
`(1,2,2)`. Convolutions mix inter-slice information without reducing Z.
The standalone baseline defaults remain unchanged.

| Stage | Shape `[B,C,Z,H,W]` at the reference configuration | MERIT operation |
| --- | --- | --- |
| Input | `[B,1,11,128,128]` | Per-sample slab-local master |
| Encoder 1 | `[B,32,11,32,32]` | EPA |
| Encoder 2 | `[B,64,11,16,16]` | EPA |
| Encoder 3 | `[B,128,11,8,8]` | EPA |
| Bottleneck | `[B,256,11,4,4]` | TEP before decoder5 |
| decoder3 | `[B,32,11,32,32]` | MER before decoder2 |
| Full decoder | `[B,16,11,128,128]` | PCG presence head |
| Output | `[B,2,11,128,128]` | PCG foreground correction |

The input must match the declared `img_size`; H/W must be divisible by 32.
The model does not resize, pad or interpolate a native slab to this grid.
Larger fixed grids require explicit configuration and substantially more memory.

### Equations and implementation mapping

Source: `src/merit_covid/models/merit_unetrpp/`.

- **TEP, `modules.py`:** Eq. 1 uses per-slice GAP and a `256->64->1` MLP with
  sigmoid salience. Eq. 2 concatenates salience with 64-dimensional relative PE
  and applies `65->16->1`, then Z-softmax with `softplus(tau0)+0.1` temperature.
  Eq. 3 forms a weighted prototype. Eq. 4 uses slice queries and prototype
  keys/values for spatial attention plus residual. No extra residual distance
  decay is introduced.
- **MER, `modules.py`:** Eq. 5 queries only the master and uses nonmaster context.
  PE affects keys, not content values. Attention dimension is 64 with projection
  `64->32`. Eq. 6-7 subtract normalized XY squared-distance bias. Eq. 8 normalizes
  jointly across all context slice/pixel keys. Eq. 9 applies a per-position
  `64->16->1` gate and `f_m + beta*g*delta`. Other slices remain unchanged inside
  MER. Query chunks of 128 still attend all keys; this is not local attention.
- **PCG, `modules.py`:** Eq. 10 uses spatial GAP and `16->8->1` with final bias
  `log(0.7/0.3)`. Eq. 11-12 clamp presence and add scaled log-odds only to the
  foreground logit. Background/raw logits are unchanged; no hard presence
  threshold erases a prediction.
- **`model.py`:** returns `MERITOutput(logits, presence, salience, raw_logits)`.
  Disabled TEP/PCG heads return `None` for their corresponding auxiliary output.
  All eight TEP/MER/PCG enable/disable combinations are tested.
- **`adapter.py`:** selects master logits and returns `fg-bg [B,1,H,W]`.
  Its sigmoid equals foreground softmax over the two logits. It does not
  threshold or read GT.

Default additional parameter counts match the paper's Table 3:

| Module | Parameters |
| --- | ---: |
| TEP | 279,731 |
| MER | 9,252 |
| PCG | 145 |
| Total | 289,128 |

Q/K/V/output projections have no bias; MLPs do. Position encodings are fixed
sin/cos. Query chunking limits temporary score matrices but autograd may retain
activations from multiple chunks.

### Explicit implementation choices

Matching formulas/counts does not establish exact source or result reproduction.

- MLP activation: GELU. PE: alternating sin/cos with base 10000.
- Initialization: `tau0=0`, softplus-positive PE/XY coefficients initialized
  from 1, and `beta=0.1`. These values are not fully specified by the PDF.
- Segmentation: foreground Dice with smoothing `1e-5` plus two-class CE,
  computed per valid `(batch,slice)` and Gaussian-weighted.
- Index Gaussian sigma defaults to 2 slices. Physical weighting requires a
  separate explicit `sigma_mm`; the units are not interchangeable.
- Presence uses probability BCE with configurable positive-class weight,
  default 1, under the Gaussian weighting.
- Alignment defaults to MSE between presence and maximum valid-pixel foreground
  probability from the two **raw** logits. `top_k_mean` is an optional ablation,
  not a silent replacement. Pooling/divergence details need author-source
  confirmation for exact reproduction.
- Salience uses BCE on valid slices. Presence/salience targets are `any` over
  valid pixels of each true slice target.
- Auxiliary weights: presence 0.2, alignment 0.05, salience 0.03. Disabled heads
  contribute zero without creating unused auxiliary heads.
- CUDA probability BCE explicitly runs FP32 outside autocast. Casting to
  `float()` alone does not bypass the CUDA autocast BCE prohibition.

Physical sampling, actual-mm PE, physical Gaussian weighting and unique duplicate
supervision are separately configured research choices. In `index` position
mode, the original relative-index behavior is retained. Physical PE uses
`actual_mm / position_unit_mm` and requires offsets with no silent fallback.

### Model API and warmup

```python
import torch
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss

model = MERITUNETRPP(
    img_size=(11, 32, 32), feature_size=2, depths=(1, 1, 1, 1),
)
image = torch.randn(2, 1, 11, 32, 32)
masters = torch.tensor([2, 9])  # Local positions, not CT volume indices.
target = torch.zeros(2, 1, 11, 32, 32)
model.set_epoch(0)
output = model(image, masters)
losses = MERITLoss()(output, target, masters)
losses["total"].backward()
```

This compact configuration is for software checks, not a published experiment.
A newly constructed model has PCG ramp 1; the trainer calls `set_epoch` each
epoch. With default warmup 30: epochs 0/15/30 give ramp 0/0.5/1. Ramp is saved
in the state dict. Forward does not change epoch or train/eval mode.

The master adapter accepts exactly one of `image`, `image_slab` or `image_patch`.
Slab input defaults to the center; other conventions require `master_index`.
Never use volume-level `master_slice_index` as a slab position.
Training calls the model directly to retain all four loss inputs.

### Checks and supported execution

```powershell
py scripts/model_smoke.py --model merit --backward
py scripts/model_smoke.py --model merit --config configs/merit_unetrpp.yaml
py -m pytest tests/test_merit_unetrpp.py tests/test_merit_pipeline.py -q
```

The YAML smoke configuration runs the reference model; the no-config example is
compact. Neither model-only smoke command trains a clinical model.

The recommended data path is `CTSlabDataset` with true `mask_slab` and
`valid_region`. MERIT trains on all valid slab slices but reconstructs only
one master prediction per sample. Use `engine.slab_inference.predict_slab_case`
for complete CT-only inference, not the standalone one-channel patch helper.
Full-slab patch experiments require their own explicit master and aggregation
policy and must not be conflated with the default master-volume protocol.

Independent equation checks, gradient/AMP checks, padding, warmup, ablations,
physical interfaces and geometry integration are covered by tests. See
[validation](validation.md) for measured results and unresolved research issues.

---

## Standalone UNETR++ baseline

The repository packages the Synapse 3D UNETR++ architecture, separately from
MERIT. It has model-only configuration, master-output adapter, synthetic
forward/backward tests and native patch inference. It has no dedicated baseline
trainer, trained COVID checkpoint or segmentation-performance result.

### Source and fidelity boundary

Pinned upstream: `Amshaker/unetr_plus_plus`, commit
`dcba1c6e8179b77e2ec72b5f4daced8926259f01`.
The chosen variant is Synapse throughout; Lung/ACDC/Tumor are not interchangeable
files. At this commit the Lung variant differs in E/F sharing and residual
blocks. Choosing Synapse is an integration baseline, not evidence that it is
better for COVID.

Packaged [LICENSE](../src/merit_covid/models/unetrpp/LICENSE),
[NOTICE](../src/merit_covid/models/unetrpp/NOTICE) and
[upstream.json](../src/merit_covid/models/unetrpp/upstream.json) preserve
Apache-2.0 attribution, source hashes and modifications.

### Network and shapes

```text
input -> patch stem -> four EPA encoder stages -> token-to-feature reshape
      -> decoder5/4/3 with additive encoder skips
      -> decoder2 with full-resolution convolutional skip
      -> main logits (+ two lower-resolution heads with deep supervision)
```

Reference input: `[B,1,64,128,128]`. This is an explicitly declared network
grid, not an instruction to resize the dataset.

| Stage | Shape, B=1 |
| --- | --- |
| Full-resolution convolutional skip | `[1,16,64,128,128]` |
| Encoder 1, stem stride `(2,4,4)` | `[1,32,32,32,32]` |
| Encoder 2 | `[1,64,16,16,16]` |
| Encoder 3 | `[1,128,8,8,8]` |
| Encoder 4 | `[1,256,4,4,4]` |
| Flattened encoder 4 | `[1,64,256]` |
| Main binary head | `[1,1,64,128,128]` |
| Auxiliary heads | `[1,1,32,32,32]` and `[1,1,16,16,16]` |

EPA consumes `[B,N,C]` and combines channel/spatial attention with shared Q/K.
Synapse shares token projections E/F. Each branch projects to C/2 before
concatenation. Learned positional embeddings, LayerNorm, EPA residual scale
initialized to 1e-6 and one residual convolutional block follow upstream.
Decoder skips are added, not concatenated.

`do_ds=True` returns `[main, auxiliary_1, auxiliary_2]`; `False` returns the
main tensor. All outputs are raw logits. Model forward never sigmoid/softmax,
threshold or automatically upsample auxiliary outputs.

### Input constraints

A model accepts exactly its configured `img_size` as floating `[B,C,D,H,W]`.
Default topology requires D divisible by 16 and H/W divisible by 32; the
deepest grid needs at least two normalization positions. Channels/dims/head
and GroupNorm compatibility are validated.

Token projections and learned PE depend on the fixed grid. Equal voxel counts
with different shapes are not interchangeable, and changing the grid can
invalidate checkpoint parameter shapes. `dims` must be
`feature_size * (2,4,8,16)` and `hidden_size` the final dimension.
Encoder `depths` do not change the three-block decoder depths; decoder heads
remain 4 as upstream.

The default native slab `[B,1,11,512,512]` does not satisfy this topology.
Use [native patch preparation](data_pipeline.md#native-patch-input-and-unified-volume-output), not fake reshape/padding of
an 11-slice input into a different anatomical volume.
The opt-in `preserve_z` mode is used by the separate MERIT subclass; default
`False` preserves standalone Synapse strides/state-dict shapes.

### Code map and packaging adjustments

| Local file | Responsibility / source |
| --- | --- |
| `model.py` | Constructor, fixed-grid checks and complete forward |
| `encoder.py` | Stem, four stages and hidden states |
| `epa.py`, `transformer.py` | Upstream `synapse/transformerblock.py` |
| `decoder.py` | Upsampling and additive skip blocks |
| `blocks.py` | Only used convolution helpers from `dynunet_block.py` |
| `adapter.py` | Separate binary master-logit selection |

Packaging replaces the nnU-Net `SegmentationNetwork` superclass with
`nn.Module`, uses internal imports, Torch's truncated-normal initializer instead
of timm, and einops directly. Constructor defaults are made consistent,
`in_channels` reaches the stem, and token counts derive from declared
`img_size`. Reference-grid values match upstream constants.

EPA equations, E/F sharing, flatten/reshape order, residuals, additive skips,
heads and deep-supervision resolutions are retained. Unused upstream
`pos_embed/dropout_rate` constructor arguments are not exposed as misleading
controls. Encoder/decoder attention dropout remains 0.15 and convolution dropout
0.1. No upstream trainer or sliding-window framework is vendored.

### Usage and checks

Install `.[unetrpp,test]`. The extra declares Torch >=2.4.1, MONAI 1.5.2 and
einops 0.8.2; upstream's historical environment is not a required downgrade.

```python
import torch
from merit_covid.models.unetrpp import UNETRPP

# Compact software check, not a clinical experiment.
model = UNETRPP(img_size=(32,32,32), feature_size=2, do_ds=False).eval()
with torch.inference_mode():
    logits = model(torch.randn(1,1,32,32,32))
assert logits.shape == (1,1,32,32,32)
```

```powershell
py scripts/model_smoke.py --model unetrpp --backward
py scripts/model_smoke.py --model unetrpp --config configs/unetrpp_baseline.yaml
py scripts/unetrpp_real_input.py
py -m pytest tests/test_unetrpp.py tests/test_unetrpp_input.py tests/test_prediction_contract.py -q
```

The model YAML is not a `merit_covid.config.load_config` dataset config.
With the full binary configuration there are 42,951,355 parameters.

`UNETRPPAdapter` returns `[B,1,H,W]` master logits and preserves autograd/caller
mode. It accepts exactly one of `image` or `image_slab`. Slab input defaults to
D//2; `image` requires a per-sample integer `master_index`. Volume-level
`master_slice_index` is never used as a local position. GT/metadata are not
read. Deep supervision selects only `output[0]`; direct model calls are needed
for auxiliary training losses. Native patch inference calls the 3D model
directly, not this master adapter.

### Verification and remaining work

Stored [upstream verification](../analysis/unetrpp/verification.json) records
strict state-dict correspondence and elementwise agreement of all three
reference-grid outputs against the pinned source in the same dependency
environment. This is software fidelity evidence, not published TMI/COVID result
reproduction or a guarantee across all MONAI/Torch versions.

The optional checker requires the separately acquired pinned source files:

```powershell
py tools/verify_unetrpp_upstream.py --source-dir references/unetrpp-pinned --report outputs/upstream_check.json
```

Reference clones are ignored and are unnecessary for ordinary imports/tests.
The checker validates source hashes before applying the documented import,
superclass and initializer substitutions. It does not download checkpoints.

A baseline research run still needs explicit loss/deep-supervision targets,
optimizer/schedule, valid-padding masking, training budget, checkpoint selection
and native-volume test evaluation. Use real 3D GT, not repeated master masks.
The MERIT trainer is not silently a standalone-baseline trainer.
See [validation](validation.md) for current evidence and limits.
