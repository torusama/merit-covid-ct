"""Authoritative Duong pipeline inside the main package; lazy torch API."""
from importlib import import_module
__all__ = ['PreprocessConfig', 'CTSlabDataset', 'FoldDataLoaders', 'build_fold_loaders', 'make_dataloader']

def __getattr__(name):
    module = {'PreprocessConfig': 'config', 'CTSlabDataset': 'dataset', 'FoldDataLoaders': 'dataloader', 'build_fold_loaders': 'dataloader', 'make_dataloader': 'dataloader'}.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(f'{__name__}.{module}'), name)
