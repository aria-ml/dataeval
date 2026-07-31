"""Configuration base classes and mixins for evaluators."""

__all__ = [
    "ClusterConfigMixin",
    "EvaluatorConfig",
]

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from dataeval.protocols import FeatureExtractor

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
    batch_size: int | None = None
    cluster_algorithm: Literal["kmeans", "hdbscan"] = _DEFAULT_CLUSTER_ALGORITHM
    n_clusters: int | None = _DEFAULT_CLUSTER_N_CLUSTERS
