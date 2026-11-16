"""Compare the frozen factorized MER with original MER on the actual Kaggle GPU."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
import types

import torch

from merit_covid.engine.activation_checkpointing import checkpointed_mer
from merit_covid.engine.factorized_attention import factorized_mer
from merit_covid.models.merit_unetrpp import MasterEvidenceRouting


def check(device="cuda"):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable")
    results = []
    for amp in ((False, True) if device.type == "cuda" else (False,)):
        for masked in (False, True):
            torch.manual_seed(813)
            reference = MasterEvidenceRouting(4, attention_dim=8, gate_hidden=3, query_chunk_size=5,
                                              position_mode="physical_mm").to(device).train()
            candidate = deepcopy(reference)
            reference.forward = types.MethodType(checkpointed_mer, reference)
            candidate.forward = types.MethodType(factorized_mer, candidate)
            image = torch.randn(2, 4, 3, 4, 6).to(device)
            offsets = torch.tensor([[0., 2.5, 7.], [-8., -3., 0.]])
            mask = torch.tensor([[True, True, False], [False, False, True]]) if masked else None
            tensors, outputs = [], []
            for model in (reference, candidate):
                x = image.clone().requires_grad_()
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    y = model(x, [0, 2], offsets, mask)
                    loss = y.square().mean()
                loss.backward()
                tensors.append(x)
                outputs.append(y)
            rtol, atol = (5e-3, 2e-6) if amp else (1e-4, 1e-7)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=rtol, atol=atol)
            torch.testing.assert_close(tensors[0].grad, tensors[1].grad, rtol=rtol, atol=atol)
            max_parameter_error = 0.0
            for (name, a), (other, b) in zip(reference.named_parameters(), candidate.named_parameters()):
                if name != other or a.grad is None or b.grad is None or not torch.isfinite(b.grad).all():
                    raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
                torch.testing.assert_close(a.grad, b.grad, rtol=rtol, atol=atol, msg=name)
                max_parameter_error = max(max_parameter_error, float((a.grad - b.grad).abs().max()))
            if candidate.distance_scale_raw.grad.abs() == 0:
                raise FloatingPointError("Distance-scale gradient is zero")
            if masked:
                torch.testing.assert_close(outputs[1][1, :, 2], image[1, :, 2], rtol=0, atol=0)
            results.append(dict(amp=amp, masked=masked, status="PASS",
                max_output_abs_error=float((outputs[0] - outputs[1]).abs().max()),
                max_input_grad_abs_error=float((tensors[0].grad - tensors[1].grad).abs().max()),
                max_parameter_grad_abs_error=max_parameter_error))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    result = dict(status="FAIL", torch_version=torch.__version__, cuda_version=torch.version.cuda)
    try:
        result.update(hardware=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                      tests=check("cuda"), status="PASS")
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
    finally:
        result["seconds"] = time.perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Factorized GPU correctness guard:", result["status"], flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
