"""Original Synapse UNETR++ architecture and optional MERGE output adapter.

Install model dependencies with: pip install -e ".[unetrpp]"
Importing merit_covid or its NumPy pipeline does not import this package.
"""

from .adapter import UNETRPPAdapter
from .model import UNETRPP

__all__ = ["UNETRPP", "UNETRPPAdapter"]
