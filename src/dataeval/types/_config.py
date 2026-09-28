"""Configuration base classes and mixins for evaluators."""

__all__ = [
    "ClusterConfigMixin",
    "EvaluatorConfig",
]

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from dataeval.protocols import FeatureExtractor

# Bounded floats for Config fields. Each reads as ``float`` to a type checker, and pydantic checks the
# bound and writes it into the JSON Schema. Counts use pydantic's ``PositiveInt`` and ``NonNegativeInt``.
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
"""A share or score in [0, 1]."""
OpenUnitInterval = Annotated[float, Field(gt=0.0, lt=1.0)]
"""A p-value threshold or a fraction in (0, 1), where either end would decide nothing."""
Percentage = Annotated[float, Field(ge=0.0, le=100.0)]
"""A percentile in [0, 100]."""

# Default values for ClusterConfigMixin
_DEFAULT_CLUSTER_ALGORITHM: Literal["kmeans", "hdbscan"] = "hdbscan"
_DEFAULT_CLUSTER_N_CLUSTERS: int | None = None


class EvaluatorConfig(BaseModel):
    """Base configuration class for all evaluators."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ClusterConfigMixin(BaseModel):
    """Configuration mixin for evaluators that use clustering."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    extractor: FeatureExtractor | None = None
    batch_size: PositiveInt | None = None
    cluster_algorithm: Literal["kmeans", "hdbscan"] = _DEFAULT_CLUSTER_ALGORITHM
    n_clusters: PositiveInt | None = _DEFAULT_CLUSTER_N_CLUSTERS
