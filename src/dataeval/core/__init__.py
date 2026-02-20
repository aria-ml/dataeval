"""Core stateless functions for performing dataset, metadata and model evaluation."""

__all__ = [
    "BERResult",
    "CalculationResult",
    "ClusterResult",
    "ClusterStats",
    "CompletenessResult",
    "CoverageResult",
    "DivergenceResult",
    "FeatureDistanceResult",
    "LabelErrorResult",
    "LabelParityResult",
    "LabelStatsResult",
    "MSTResult",
    "MutualInfoResult",
    "NullModelMetrics",
    "NullModelMetricsResult",
    "ParityResult",  # type: ignore - experimental
    "RankResult",
    "ber_knn",
    "ber_mst",
    "calculate_ratios",
    "calculate_stats",
    "cluster",
    "completeness",
    "compute_cluster_stats",
    "compute_neighbors",
    "coverage_adaptive",
    "coverage_naive",
    "dhash",
    "dhash_d4",
    "divergence_fnn",
    "divergence_mst",
    "feature_distance",
    "factor_deviation",
    "factor_predictors",
    "label_errors",
    "label_parity",
    "label_stats",
    "minimum_spanning_tree",
    "mutual_info",
    "mutual_info_classwise",
    "nullmodel_accuracy",
    "nullmodel_fpr",
    "nullmodel_precision",
    "nullmodel_metrics",
    "nullmodel_recall",
    "parity",  # type: ignore - experimental
    "phash",
    "phash_d4",
    "rank_hdbscan_complexity",
    "rank_hdbscan_distance",
    "rank_knn",
    "rank_kmeans_distance",
    "rank_kmeans_complexity",
    "rank_result_class_balanced",
    "rank_result_stratified",
    "xxhash",
    "uap",  # type: ignore - experimental
]

from typing import Any

from dataeval.core._ber import BERResult, ber_knn, ber_mst
from dataeval.core._calculate_ratios import calculate_ratios
from dataeval.core._calculate_stats import CalculationResult, calculate_stats
from dataeval.core._clusterer import ClusterResult, ClusterStats, cluster, compute_cluster_stats
from dataeval.core._completeness import CompletenessResult, completeness
from dataeval.core._coverage import CoverageResult, coverage_adaptive, coverage_naive
from dataeval.core._divergence import DivergenceResult, divergence_fnn, divergence_mst
from dataeval.core._feature_distance import FeatureDistanceResult, feature_distance
from dataeval.core._hash import dhash, dhash_d4, phash, phash_d4, xxhash
from dataeval.core._label_errors import LabelErrorResult, label_errors
from dataeval.core._label_parity import LabelParityResult, label_parity
from dataeval.core._label_stats import LabelStatsResult, label_stats
from dataeval.core._metadata_insights import factor_deviation, factor_predictors
from dataeval.core._mst import MSTResult, compute_neighbors, minimum_spanning_tree
from dataeval.core._mutual_info import MutualInfoResult, mutual_info, mutual_info_classwise
from dataeval.core._nullmodel import (
    NullModelMetrics,
    NullModelMetricsResult,
    nullmodel_accuracy,
    nullmodel_fpr,
    nullmodel_metrics,
    nullmodel_precision,
    nullmodel_recall,
)
from dataeval.core._rank import (
    RankResult,
    rank_hdbscan_complexity,
    rank_hdbscan_distance,
    rank_kmeans_complexity,
    rank_kmeans_distance,
    rank_knn,
    rank_result_class_balanced,
    rank_result_stratified,
)

_EXPERIMENTAL: dict[str, tuple[str, str]] = {
    "parity": ("dataeval.core._parity", "parity"),
    "ParityResult": ("dataeval.core._parity", "ParityResult"),
    "uap": ("dataeval.core._uap", "uap"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPERIMENTAL:
        from dataeval._experimental import _lazy_import_with_warning

        module_path, attr_name = _EXPERIMENTAL[name]
        return _lazy_import_with_warning(module_path, attr_name, f"dataeval.core.{name}", "experimental")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
