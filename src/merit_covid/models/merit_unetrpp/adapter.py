"""MERIT's two-channel output -> the existing one-channel master-logit contract."""

import torch
from torch import nn

from .model import MERITUNETRPP
from .modules import master_positions


class MERITUNETRPPAdapter(nn.Module):
    input_kind = "slab_3d"

    def __init__(self, model):
        super().__init__()
        if not isinstance(model, MERITUNETRPP):
            raise TypeError("MERITUNETRPPAdapter requires a MERITUNETRPP model")
        self.model = model

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise ValueError("Expected a collated batch dictionary")
        keys = [key for key in ("image", "image_slab", "image_patch") if key in batch]
        if len(keys) != 1:
            raise ValueError("Provide exactly one of image, image_slab or image_patch")
        image = batch[keys[0]]
        self.model.validate_input(image)
        indices = batch.get("master_index")
        if indices is None:
            if keys[0] != "image_slab":
                raise ValueError("image/image_patch batches require explicit master_index")
            indices = [image.shape[2] // 2] * image.shape[0]
        master = master_positions(indices, image.shape[0], image.shape[2], image.device)
        output = self.model(image, master, batch.get("z_offsets_mm"))
        # sigmoid(fg-bg) == softmax([bg,fg])[fg], retaining all gradients.
        binary_logits = output.logits[:, 1:2] - output.logits[:, :1]
        return binary_logits[torch.arange(image.shape[0], device=image.device), :, master]

    def predict_logits(self, batch):
        return self(batch)
