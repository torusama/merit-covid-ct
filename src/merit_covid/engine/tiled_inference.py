"""CT-only tile inference; probabilities stitched before original-volume metrics."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from merit_covid.data.pipeline.dataloader import make_dataloader
from merit_covid.data.pipeline.metrics import (aggregate_results, volume_metrics, lesion_slice_metrics,
                                             probabilities, validate_threshold)
from merit_covid.data.pipeline.tiled import TiledCase, TiledCTSlabDataset, reconstruct_gt, validate_tile_policy
from merit_covid.models.merit_unetrpp import MERITUNETRPPAdapter
from .prediction import VolumePrediction


class TileProbabilityReconstructor:
    """Strict fixed 4x4, non-overlapping [Y,X] assembly for one original case."""
    def __init__(self, case):
        self.case = case
        self.volume = np.empty((case.record["depth"], 512, 512), dtype=np.float32)
        self.seen = np.zeros((case.record["depth"], 16), dtype=bool)

    def add(self, case_id, tile_id, master_z, crop, probability):
        if (case_id != self.case.case_id or type(tile_id) is not int or not 0 <= tile_id < 16
                or type(master_z) is not int or not 0 <= master_z < len(self.seen)):
            raise ValueError("Wrong case/tile/master Z in probability reconstruction")
        row, col = divmod(tile_id, 4)
        expected = dict(x0=col*128, x1=(col+1)*128, y0=row*128, y1=(row+1)*128)
        if crop != expected or self.seen[master_z, tile_id]:
            raise ValueError("Wrong XY crop or duplicate/overlapping tile prediction")
        p = probabilities(probability)
        if p.shape != (128, 128) or not np.issubdtype(p.dtype, np.floating):
            raise ValueError("Tile probability must be [128,128]; never resize/threshold tiles")
        self.volume[master_z, expected["y0"]:expected["y1"], expected["x0"]:expected["x1"]] = p
        self.seen[master_z, tile_id] = True

    def finish(self):
        if not self.seen.all():
            raise ValueError(f"Missing tile predictions/gaps: {int((~self.seen).sum())}")
        return self.volume


def predict_tiled_case(model, case, preprocess, *, batch_size=1, num_workers=0):
    if not isinstance(case, TiledCase) or model.training:
        raise ValueError("Tiled inference needs a TiledCase and model.eval()")
    validate_tile_policy(preprocess)
    if tuple(model.img_size) != (11, 128, 128):
        raise ValueError("Tiled inference requires [11,128,128] model grid")
    dataset = TiledCTSlabDataset([case], preprocess, require_mask=False)
    loader = make_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
    adapter = MERITUNETRPPAdapter(model)
    assembly = TileProbabilityReconstructor(case)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            batch["image_slab"] = batch["image_slab"].to(device)
            logits = adapter.predict_logits(batch)
            if (logits.shape != (len(batch["image_slab"]), 1, 128, 128)
                    or not logits.is_floating_point() or not torch.isfinite(logits).all()):
                raise ValueError("MERIT tile master logits must be finite floating [B,1,128,128]")
            values = logits.float().sigmoid().cpu().numpy()
            for i, meta in enumerate(batch["metadata"]):
                if meta["full_volume_metadata"] != case.metadata or meta["case_id"] != batch["case_id"][i]:
                    raise ValueError("Wrong full-volume metadata during tile inference")
                assembly.add(batch["case_id"][i], int(batch["tile_id"][i]), int(batch["master_slice_index"][i]),
                             meta["xyz_crop"], values[i, 0])
    metadata = deepcopy(case.metadata)
    metadata["prediction_policy"] = dict(model="MERIT-UNETR++", activation="sigmoid(fg-bg)",
        input=asdict(preprocess), output="stitch sixteen probabilities per master; no resize; threshold only after full-volume restoration",
        whole_slice_presence="UNSUPPORTED: tile-local presence is not calibrated whole-slice presence")
    return VolumePrediction(case.case_id, assembly.finish(), metadata)


def evaluate_tiled_model(model, cases, preprocess, *, threshold=.5, include_hd95=True,
                         batch_size=1, num_workers=0, identity=None):
    validate_threshold(threshold)
    validate_tile_policy(preprocess)
    if not cases or any(not isinstance(c, TiledCase) for c in cases):
        raise ValueError("Cannot mix native and tiled original cases")
    was_training = model.training; model.eval(); rows = []
    try:
        for case in cases:
            prediction = predict_tiled_case(model, case, preprocess, batch_size=batch_size, num_workers=num_workers)
            # No GT is opened by the inference function. Labels enter only here.
            canonical_gt, original_gt = reconstruct_gt(case, assume_mm=preprocess.assume_mm)
            metrics = volume_metrics(prediction.to_original_xyz(), original_gt, threshold,
                                     case.metadata["original_affine_mm"], include_hd95)
            metrics["lesion_slice_detail"] = lesion_slice_metrics(prediction.probability_zyx, canonical_gt, threshold, None)
            rows.append(dict(case_id=case.case_id, group_id=case.group_id, **metrics,
                lesion_retention=dict(resize_applied=False, lost_positive_slices=[], lost_components_by_slice=[])))
    finally:
        model.train(was_training)
    return {**aggregate_results(rows, identity), "threshold": threshold,
        "threshold_policy": "fixed a priori in frozen experiment config, not test-tuned",
        "evaluation_grid": "full original NIfTI XYZ reconstructed from 16 tiles; one row per original volume",
        "whole_slice_presence_status": "UNSUPPORTED (tile-local auxiliary targets only)", "preprocess": asdict(preprocess)}
