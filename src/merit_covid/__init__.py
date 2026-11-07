"""Data and evaluation foundations; no research model is implemented."""

__version__ = "0.1.0"

# Stable Duong batch API hosted in the main package; loaded only on request.
def __getattr__(name):
    if name in ('PreprocessConfig','CTSlabDataset','FoldDataLoaders','build_fold_loaders','make_dataloader'):
        from .data import pipeline
        return getattr(pipeline,name)
    raise AttributeError(name)
