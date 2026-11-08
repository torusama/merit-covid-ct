"""Optional master-logit view; never resizes data or changes UNETR++ forward."""

import operator

import torch
from torch import nn

from .model import UNETRPP


class UNETRPPAdapter(nn.Module):
    """Implements models.base.ModelAdapter for ALREADY compatible 3D inputs.

    This bridges output semantics only. It cannot bridge the native D=11 slab
    to the model's fixed token grid. No GT is read or repeated along depth.
    predict_logits retains gradients and the caller's train/eval mode.
    """

    input_kind = "slab_3d"

    def __init__(self, model: UNETRPP):
        super().__init__()
        if not isinstance(model, UNETRPP):
            raise TypeError("UNETRPPAdapter requires a UNETRPP model")
        if model.num_classes != 1:
            raise ValueError("The MERGE binary master-slice contract requires out_channels=1")
        self.model = model

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise ValueError("Expected a collated MERGE batch dictionary")
        keys = [key for key in ("image", "image_slab") if key in batch]
        if len(keys) != 1:
            raise ValueError("Provide exactly one of image or image_slab")
        image = batch[keys[0]]
        self.model.validate_input(image)
        count, _, depth, _, _ = image.shape
        indices = batch.get("master_index")
        if indices is None:
            if keys[0] == "image":
                raise ValueError("A main-API batch requires master_index (position inside slab)")
            indices = [depth // 2] * count
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1 or indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
                raise ValueError("master_index must be a vector of integer slab positions")
            indices = indices.tolist()
        if not isinstance(indices, (list, tuple)) or len(indices) != count:
            raise ValueError("One integer master_index is required per batch element")
        positions = []
        for value in indices:
            try:
                if isinstance(value, bool):
                    raise TypeError
                index = operator.index(value)
            except TypeError:
                raise ValueError("master_index contains a non-integer slab position") from None
            if not 0 <= index < depth:
                raise ValueError("master_index is outside the slab")
            positions.append(index)
        output = self.model(image)
        primary = output[0] if isinstance(output, list) else output
        # master_slice_index belongs to the original CT; never use it here.
        return primary[torch.arange(count, device=primary.device), :,
                       torch.tensor(positions, device=primary.device)]

    def predict_logits(self, batch):
        return self(batch)
