# Compatibility imports

`ct_pipeline` forwards imports to the installed `merit_covid.data.pipeline` package.
There is no second preprocessing implementation here.

Start with [the main README](../README.md), [source map](../docs/architecture.md)
and [merge report](../docs/merge_report.md).

Legacy `from ct_pipeline import PreprocessConfig, build_fold_loaders` still works.
New code uses `from merit_covid import PreprocessConfig, build_fold_loaders`.
