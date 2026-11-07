# Original MERIT-UNETR++ integration boundary

No original MERIT source or complete TEP/MER/PCG specification is supplied here.
This folder is the integration location, not an implemented model.

Obtain the supervisor's authorized source and identify its commit, license, paper,
checkpoint provenance, preprocessing, tensor layouts and auxiliary losses. Verify
the UNETR++ backbone and TEP/MER/PCG definitions against that source. Implement a
`ModelAdapter` exposing master logits `[B,1,H,W]` from slabs `[B,1,D,H,W]`.
Confirm whether its depth and patch constraints accept 11 slices; document any
required adapter transformation instead of silently changing the architecture.

U-Net 2D and MADGNet adapters select the master via `master_slice_input`. UNETR++
requires an explicit master-output mapping. nnU-Net needs its own planning and
export adapter; it is not assumed to share a training loop. All adapters return
predictions on the recorded preprocessing grid for common reconstruction/evaluation.

There is no dummy segmentation network, Reliability Gate or new routing module.
Spacing metadata is supplied by the data contract for provenance and future work;
it does not imply that the original MER consumes millimetre offsets.
