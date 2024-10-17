"""
Update strategies inform how the drift detector classes update the reference data when monitoring for drift.
"""

from dataeval._internal.detectors.drift.base import LastSeenUpdate, ReservoirSamplingUpdate

__all__ = ["LastSeenUpdate", "ReservoirSamplingUpdate"]
