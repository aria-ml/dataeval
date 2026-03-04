"""
Evaluate data completeness and coverage in latent embedding space.

Scope evaluators assess whether a dataset adequately spans its embedding
space, identifying gaps in coverage and prioritizing new data for labeling.
"""

__all__ = ["Prioritize", "PrioritizeOutput"]

from ._prioritize import Prioritize, PrioritizeOutput
