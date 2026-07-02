"""
Evaluate data completeness and coverage of a dataset's label and embedding space.

Scope evaluators assess whether a dataset adequately spans the space it is meant to
cover, identifying gaps and prioritizing new data for labeling — across an ontology's
label space (:class:`Representation`) and a dataset's latent embedding space
(:class:`Coverage`, :class:`Prioritize`).
"""

__all__ = ["Coverage", "CoverageOutput", "Prioritize", "PrioritizeOutput", "Representation", "RepresentationOutput"]

from ._coverage import Coverage, CoverageOutput
from ._prioritize import Prioritize, PrioritizeOutput
from ._representation import Representation, RepresentationOutput
