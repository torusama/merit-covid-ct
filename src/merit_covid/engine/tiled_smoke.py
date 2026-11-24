"""Exactly one tile forward/loss/backward, no optimizer or epoch iteration."""
import torch

from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataloader import collate_slabs
from merit_covid.data.pipeline.tiled import TiledCTSlabDataset
from merit_covid.models.merit_unetrpp import MERITUNETRPP, MERITLoss
from .activation_checkpointing import configure_model_execution
from .training import resolve_device


def tile_forward_backward(config, bundle, case_id, device="cpu"):
    cases = [c for c in bundle.cases if c.case_id == case_id]
    if len(cases) != 1:
        raise ValueError("Choose one existing original case ID for io-smoke")
    torch.set_num_threads(config["training"]["threads"])
    torch.manual_seed(config["training"]["seed"])
    device = resolve_device(device)
    dataset = TiledCTSlabDataset(cases, PreprocessConfig(**config["preprocess"]))
    batch = collate_slabs([dataset[0]])
    model = configure_model_execution(MERITUNETRPP(**config["model"]).to(device), config["training"])
    model.train()  # All TEP/MER/PCG remain enabled as configured; full PCG ramp.
    model.set_pcg_ramp(1.)
    criterion = MERITLoss(**config["loss"])
    amp = config["training"]["amp"] and device.type == "cuda"
    try:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(batch["image_slab"].to(device), batch["master_index"], batch["z_offsets_mm"], batch["attention_valid"])
            losses = criterion(output, batch["mask_slab"].to(device), batch["master_index"],
                               valid_region=batch["valid_region"].to(device), z_offsets_mm=batch["z_offsets_mm"])
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise FloatingPointError("Nonfinite smoke loss")
        losses["total"].backward()
        finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        if not finite:
            raise FloatingPointError("Missing/nonfinite smoke gradients")
        return dict(status="PASS", case_id=case_id, tile_id=0, master_z=0,
            input_shape=list(batch["image_slab"].shape), mask_shape=list(batch["mask_slab"].shape),
            logit_shape=list(output.logits.shape), losses={k:float(v.detach()) for k,v in losses.items()},
            finite_gradients=True, device=str(device), optimizer_updates=0, real_training=False,
            preprocessing="pass-through precomputed V3", gpu_benchmark=False)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError("Insufficient memory for unchanged Tiling128 model; no model/input reduction was applied") from exc
