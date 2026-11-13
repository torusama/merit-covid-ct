# MERIT-UNETR++ model

This is a paper-derived reimplementation, not the authors' source.
The integrated trainer is described in [merit_pipeline.md](merit_pipeline.md).
No trained COVID checkpoint or replicated research result is provided.

## Source and task transfer

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

## Backbone and insertion points

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

## Equations and implementation mapping

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

## Explicit implementation choices

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

## Model API and warmup

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

## Checks and supported execution

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
