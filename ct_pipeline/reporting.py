"""Compatibility alias; implementation moved into merit_covid.data.pipeline."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("merit_covid.data.pipeline.reporting")
