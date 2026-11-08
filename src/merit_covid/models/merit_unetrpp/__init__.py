"""Paper-derived MERIT-UNETR++ implementation; optional UNETR++ dependencies."""

from .adapter import MERITUNETRPPAdapter
from .loss import MERITLoss
from .model import MERITOutput, MERITUNETRPP
from .modules import MasterEvidenceRouting, PresenceCommitmentGate, TrajectoryEvidencePrototype

__all__ = ["MERITUNETRPP", "MERITUNETRPPAdapter", "MERITOutput", "MERITLoss",
           "TrajectoryEvidencePrototype", "MasterEvidenceRouting", "PresenceCommitmentGate"]
