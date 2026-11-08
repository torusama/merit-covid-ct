# Standalone UNETR++ baseline

The repository packages the Synapse 3D UNETR++ architecture, separately from
MERIT. It has model-only configuration, master-output adapter, synthetic
forward/backward tests and native patch inference. It has no dedicated baseline
trainer, trained COVID checkpoint or segmentation-performance result.

## Source and fidelity boundary

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

## Network and shapes

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

## Input constraints

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
Use [native patch preparation](unetrpp_input.md), not fake reshape/padding of
an 11-slice input into a different anatomical volume.
The opt-in `preserve_z` mode is used by the separate MERIT subclass; default
`False` preserves standalone Synapse strides/state-dict shapes.

## Code map and packaging adjustments

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

## Usage and checks

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
py examples/unetrpp_smoke.py --backward
py examples/unetrpp_smoke.py --config configs/unetrpp_baseline.yaml
py examples/unetrpp_real_input.py
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

## Verification and remaining work

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
