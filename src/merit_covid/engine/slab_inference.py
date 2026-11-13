"""CT-only master-slab inference, assembled on the full native canonical grid."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from merit_covid.data.pipeline.dataset import CTSlabDataset
from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.data.pipeline.metrics import SliceReconstructor
from merit_covid.models.merit_unetrpp import MERITUNETRPPAdapter
from .prediction import VolumePrediction, binary_logits_to_probabilities


def predict_slab_case(model, case, preprocess, *, batch_size=1, num_workers=0):
    """One prediction for EVERY canonical master, no GT read and no thresholds.

    XY probabilities are resized before constructing the native-only unified
    contract. The original SliceReconstructor checks duplicates/missing slices.
    """
    if model.training:
        raise ValueError("Call model.eval() before inference")
    if preprocess.slab_size != model.img_size[0] or (preprocess.size_hw is not None and tuple(preprocess.size_hw) != model.img_size[1:]):
        raise ValueError("Slab preprocessing and model img_size disagree")
    dataset = CTSlabDataset([case], preprocess, require_mask=False)
    loader = make_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
    adapter = MERITUNETRPPAdapter(model)
    device = next(model.parameters()).device
    assembly, native_metadata = None, None
    with torch.inference_mode():
        for batch in loader:
            batch["image_slab"] = batch["image_slab"].to(device)
            logits = adapter.predict_logits(batch)
            if (not isinstance(logits, torch.Tensor)
                    or logits.shape != (len(batch["image_slab"]), 1, *batch["image_slab"].shape[-2:])
                    or not logits.is_floating_point() or not torch.isfinite(logits).all()):
                raise ValueError("MERIT master logits must be finite floating [B,1,H,W] before sigmoid")
            for i, metadata in enumerate(batch["metadata"]):
                if batch["case_id"][i] != case.case_id or metadata["case_id"] != case.case_id:
                    raise ValueError("Wrong case during master reconstruction")
                if native_metadata is None:
                    native_metadata = deepcopy(metadata)
                    native_metadata["processed_size_hw"] = list(metadata["native_size_hw"])
                    native_metadata["prediction_policy"] = {
                        "model": "MERIT-UNETR++", "activation": "sigmoid(fg-bg)",
                        "input": asdict(preprocess), "position_mode": model.position_mode,
                        "position_unit_mm": model.position_unit_mm,
                        "output": "one prediction per canonical master; bilinear native XY before threshold",
                    }
                    assembly = SliceReconstructor(tuple(reversed(metadata["canonical_shape_xyz"])))
                elif any(metadata[key] != native_metadata[key] for key in (
                        "canonical_affine_mm", "original_affine_mm", "canonical_shape_xyz", "original_shape_xyz")):
                    raise ValueError("Geometry changed during master reconstruction")
                probability = logits[i:i+1].float().sigmoid()
                if tuple(probability.shape[-2:]) != tuple(metadata["native_size_hw"]):
                    probability = F.interpolate(probability, size=metadata["native_size_hw"], mode="bilinear", align_corners=False)
                assembly.add(int(batch["master_slice_index"][i]), probability[0, 0].cpu().numpy())
    if assembly is None:
        raise ValueError("Empty inference dataset")
    return VolumePrediction(case.case_id, assembly.finish(), native_metadata)
