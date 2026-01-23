"""
Core stateless functions for performing dataset, metadata and model evaluation.
"""

__all__ = [
    "ber_knn",
    "ber_mst",
    "calculate",
    "calculate_ratios",
    "cluster",
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
    "parity",
    "phash",
    "phash_d4",
    "rank_hdbscan_complexity",
    "rank_hdbscan_distance",
    "rank_knn",
    "rank_kmeans_distance",
    "rank_kmeans_complexity",
    "rerank_hard_first",
    "rerank_stratified",
    "rerank_class_balance",
    "xxhash",
    "uap",
]

from dataeval.core._ber import ber_knn, ber_mst
from dataeval.core._calculate import calculate
from dataeval.core._calculate_ratios import calculate_ratios
from dataeval.core._clusterer import cluster, compute_cluster_stats
from dataeval.core._coverage import coverage_adaptive, coverage_naive
from dataeval.core._divergence import divergence_fnn, divergence_mst
from dataeval.core._feature_distance import feature_distance
from dataeval.core._hash import dhash, dhash_d4, phash, phash_d4, xxhash
from dataeval.core._label_errors import label_errors
from dataeval.core._label_parity import label_parity
from dataeval.core._label_stats import label_stats
from dataeval.core._metadata_insights import factor_deviation, factor_predictors
from dataeval.core._mst import compute_neighbors, minimum_spanning_tree
from dataeval.core._mutual_info import mutual_info, mutual_info_classwise
from dataeval.core._nullmodel import (
    nullmodel_accuracy,
    nullmodel_fpr,
    nullmodel_metrics,
    nullmodel_precision,
    nullmodel_recall,
)
from dataeval.core._parity import parity
from dataeval.core._rank import (
    rank_hdbscan_complexity,
    rank_hdbscan_distance,
    rank_kmeans_complexity,
    rank_kmeans_distance,
    rank_knn,
)
from dataeval.core._rerank import rerank_class_balance, rerank_hard_first, rerank_stratified
from dataeval.core._uap import uap
