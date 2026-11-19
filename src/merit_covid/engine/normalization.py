"""Keep trained BatchNorm statistics fixed while other model weights learn."""
import torch


def _freeze_before_forward(model, inputs):
    if model.training:
        for module in model._frozen_batchnorm_modules:
            module.eval()


def enable_frozen_batchnorm_stats(model):
    """Enforce fixed running statistics on every training forward.

    A parent ``train()`` call normally re-enables all child BatchNorm layers.
    The model hook reapplies this policy before any layer sees the next batch.
    Affine parameters retain their gradients; dropout and other layers retain
    their existing training mode. No parameters or state-dict keys are added.
    """
    if getattr(model, '_frozen_batchnorm_stats_enabled', False):
        return model
    modules = tuple(module for module in model.modules()
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    if any(not module.track_running_stats or module.running_mean is None or module.running_var is None
           for module in modules):
        raise ValueError('Freezing BatchNorm statistics requires saved running mean and variance')
    model._frozen_batchnorm_modules = modules
    model.register_forward_pre_hook(_freeze_before_forward)
    model._frozen_batchnorm_stats_enabled = True
    _freeze_before_forward(model, ())
    return model
