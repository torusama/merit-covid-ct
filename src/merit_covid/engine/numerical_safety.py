"""Retry overflowing AMP convolutions in FP32 without changing model state."""
import types
import torch


def safe_conv3d_forward(self, image):
    operation = (torch.nn.ConvTranspose3d.forward if isinstance(self, torch.nn.ConvTranspose3d)
                 else torch.nn.Conv3d.forward)
    output = operation(self, image)
    if not torch.is_autocast_enabled(image.device.type) or torch.isfinite(output).all():
        return output
    if not torch.isfinite(image).all() or not torch.isfinite(self.weight).all():
        raise FloatingPointError(f'Nonfinite convolution input/weight: {self._precision_layer_name}')
    # The failed result has no consumers and no stateful normalization ran yet.
    # Only this convolution is retried; autograd follows the FP32 result.
    with torch.autocast(device_type=image.device.type, enabled=False):
        output = operation(self, image.float())
    if not torch.isfinite(output).all():
        raise FloatingPointError(f'Convolution remains nonfinite in FP32: {self._precision_layer_name}')
    self._fp32_retry_count += 1
    if self._fp32_retry_count == 1:
        print(f'AMP convolution retried in FP32: {self._precision_layer_name}', flush=True)
    return output


def enable_safe_amp_convolutions(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)) and not hasattr(module, '_fp32_retry_count'):
            module._fp32_retry_count = 0
            module._precision_layer_name = name
            module.forward = types.MethodType(safe_conv3d_forward, module)
    return model
