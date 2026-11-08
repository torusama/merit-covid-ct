"""Forward-only patch integration; no losses, updates, checkpoints or metrics."""

from merit_covid.data.unetrpp_input import make_patch_dataloader
from merit_covid.data.unetrpp_input.tiling import integer_index
from .prediction import binary_logits_to_probabilities
from .patch_reconstruction import PatchReconstructor


def predict_patch_batch(model, batch):
    """Return probabilities [B,D,H,W]; caller must explicitly select eval mode."""
    import torch
    if model.training:
        raise ValueError("Call model.eval() before forward-only patch prediction")
    if getattr(model, "position_mode", None) is not None or hasattr(model, "pcg_ramp"):
        raise ValueError("MERIT requires designated-master slab inference; use predict_slab_case, not the UNETR++ patch path")
    image = batch["image_patch"]
    if not isinstance(image, torch.Tensor) or image.ndim != 5 or image.shape[1] != 1:
        raise ValueError("Expected collated image_patch tensor [B,1,D,H,W]")
    device = next(model.parameters(), torch.empty(0)).device
    with torch.inference_mode():
        output = model(image.to(device))
        primary = output[0] if isinstance(output, (list, tuple)) else output
        if not isinstance(primary, torch.Tensor) or primary.shape != image.shape:
            raise ValueError("Model primary logits must match [B,1,D,H,W]; auxiliary logits are not stitched")
        return binary_logits_to_probabilities(primary)


def predict_patch_case(model, dataset, case_index=0, *, batch_size=1, num_workers=0, model_name="UNETR++"):
    """One full CT -> VolumePrediction, with lazy patch batches and no GT reads.

    This is software inference, not evaluation. No threshold or metric is used.
    It deliberately refuses a Dataset configured to read ground truth.
    """
    from torch.utils.data import Subset
    if dataset.require_mask:
        raise ValueError("Use require_mask=False for CT-only prediction")
    case_index = integer_index(case_index, len(dataset.cases))
    case, plan = dataset.cases[case_index], dataset.plans[case_index]
    loader = make_patch_dataloader(Subset(dataset, dataset.case_patch_range(case_index)),
                                   batch_size=batch_size, num_workers=num_workers)
    assembly = None
    for batch in loader:
        probs = predict_patch_batch(model, batch)
        for i, metadata in enumerate(batch["metadata"]):
            if assembly is None:
                assembly = PatchReconstructor(case.case_id, plan, metadata,
                    policy={"model": model_name, "activation": "sigmoid", "selection": dataset.selection})
            assembly.add(int(batch["patch_index"][i]), probs[i], metadata=metadata)
    if assembly is None:
        raise ValueError("No patches available")
    return assembly.finish()
