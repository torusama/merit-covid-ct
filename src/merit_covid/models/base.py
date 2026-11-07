"""Torch-independent adapter contract for future 2D and 3D architectures."""

from typing import Any, Literal, Protocol


class ModelAdapter(Protocol):
    input_kind: Literal["master_2d", "slab_3d"]

    def predict_logits(self, batch: dict[str, Any]) -> Any:
        """Return master-slice logits [B,1,H,W] at the preprocessed XY grid.

        batch.image is [B,1,D,H,W]. A 2D adapter extracts each batch element's
        master_index. A 3D adapter explicitly maps its output to that master.
        Deep supervision and PCG auxiliaries remain inside the model adapter.
        No thresholds or ground-truth tensors may influence inference.
        """
        ...


def master_slice_input(batch: dict[str, Any]) -> Any:
    """Select per-sample master slices for NumPy arrays or torch tensors."""
    image = batch["image"]
    if len(image.shape) != 5 or image.shape[1] != 1:
        raise ValueError("Expected image [B,1,D,H,W]")
    indices = batch["master_index"]
    if len(indices) != image.shape[0]:
        raise ValueError("One master index required per batch element")
    masters = []
    for i, index in enumerate(indices):
        index = int(index)
        if not 0 <= index < image.shape[2]:
            raise ValueError("Invalid master index")
        masters.append(image[i:i + 1, :, index])
    if type(image).__module__.startswith("torch"):
        import torch
        return torch.cat(masters, dim=0)
    import numpy as np
    return np.concatenate(masters, axis=0)
