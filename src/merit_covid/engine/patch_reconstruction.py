"""Coordinate-based probability assembly, not a model-specific evaluator."""

from copy import deepcopy
import json

import numpy as np

from merit_covid.data.pipeline.cases import identifier
from merit_covid.data.pipeline.metrics import probabilities
from merit_covid.data.unetrpp_input.tiling import integer_index
from .prediction import VolumePrediction, validate_prediction_metadata, GEOMETRY_KEYS


def geometry_signature(metadata):
    return json.dumps({key: metadata[key] for key in GEOMETRY_KEYS}, sort_keys=True, allow_nan=False)


class PatchReconstructor:
    """Uniform arithmetic mean of PROBABILITIES over real voxels only.

    Every scheduled patch is required exactly once, in any order. Coordinates
    come from the same plan as extraction. Missing/duplicate/wrong-case patches
    are errors; zero-coverage voxels can never silently become background.
    """

    def __init__(self, case_id, plan, metadata, *, policy=None):
        identifier(case_id, "case_id")
        validate_prediction_metadata(metadata, plan.shape_zyx)
        if metadata.get("case_id", case_id) != case_id:
            raise ValueError("Reconstruction case_id disagrees with metadata")
        self.case_id, self.plan = case_id, plan
        self.metadata = deepcopy(metadata)
        self.metadata.pop("patch", None)
        self.signature = geometry_signature(metadata)
        self.metadata["prediction_policy"] = {
            **deepcopy(policy or {}), "input": plan.config.to_dict(),
            "output": {"blend": "uniform mean of probabilities", "padding_removed": True,
                       "grid": "full native-size LPS", "threshold_applied": False},
        }
        self._sum = np.zeros(plan.shape_zyx, dtype=np.float32)
        self._counts = np.zeros(plan.shape_zyx, dtype=np.uint32)
        self._seen = np.zeros(len(plan), dtype=bool)
        self._result = None

    def add(self, patch_index, probability_patch, *, metadata=None):
        if self._result is not None:
            raise ValueError("Cannot add patches after finish")
        index = integer_index(patch_index, len(self.plan))
        if self._seen[index]:
            raise ValueError(f"Duplicate patch: {index}")
        location = self.plan[index]
        if metadata is not None:
            if (metadata.get("case_id") != self.case_id
                    or metadata.get("patch") != location.to_dict()
                    or metadata.get("patch_count") != len(self.plan)
                    or geometry_signature(metadata) != self.signature
                    or metadata.get("input_policy") != self.plan.config.to_dict()):
                raise ValueError("Patch case/coordinates/geometry/policy disagree with reconstruction plan")
        patch = probabilities(probability_patch)
        if patch.shape != self.plan.config.patch_size:
            raise ValueError(f"Expected patch probability shape {self.plan.config.patch_size}, got {patch.shape}")
        self._sum[location.source_slices] += patch[location.valid_slices].astype(np.float32, copy=False)
        self._counts[location.source_slices] += 1
        self._seen[index] = True

    def coverage_summary(self):
        return {
            "expected_patches": len(self.plan), "received_patches": int(self._seen.sum()),
            "minimum_contributions": int(self._counts.min()),
            "maximum_contributions": int(self._counts.max()),
            "uncovered_voxels": int(np.count_nonzero(self._counts == 0)),
        }

    def finish(self):
        if self._result is not None:
            return self._result
        if not self._seen.all():
            raise ValueError(f"Missing patches: {np.flatnonzero(~self._seen).tolist()}")
        if np.any(self._counts == 0):
            raise ValueError("Incomplete volume coverage")
        result = np.empty(self.plan.shape_zyx, dtype=np.float32)
        np.divide(self._sum, self._counts, out=result)
        self.metadata["prediction_policy"]["coverage"] = self.coverage_summary()
        self._result = VolumePrediction(self.case_id, result, self.metadata)
        # Keep counts for diagnostics, release the probability-sum buffer.
        self._sum = None
        return self._result
