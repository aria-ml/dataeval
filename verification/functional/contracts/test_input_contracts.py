"""Verify that public API classes validate inputs and raise clear errors.

These tests verify high-level contract enforcement, not individual function
correctness (which is covered by the unit test suite under tests/).

Maps to meta repo test cases:
  - TC-3.1: Data Quality Analysis (Duplicates, Outliers input contracts)
"""

# pyright: reportArgumentType=false
# These tests pass invalid arguments on purpose.
import numpy as np
import pytest

INVALID_ARGUMENT_IDS = [
    "diversity-method",
    "outliers-threshold",
    "duplicates-flags",
    "drift-univariate-method",
    "drift-univariate-correction",
    "drift-kneighbors-k",
    "ood-kneighbors-k",
    "drift-wasserstein-ratio",
    "prioritize-method",
    "sufficiency-runs",
    "sufficiency-substeps",
    "embeddings-batch-size",
    "bovw-vocab-size",
    "resize-size",
    "resize-mode",
    "select-channels",
    "crop-region",
    "metadata-length-mismatch",
]


def _invalid_argument_cases():
    """(id, callable that must raise, argument the message must name)."""
    from dataeval import Embeddings, Metadata
    from dataeval.bias import Diversity
    from dataeval.data import Crop, Resize, SelectChannels
    from dataeval.extractors import BoVWExtractor
    from dataeval.flags import ImageStats
    from dataeval.performance import Sufficiency
    from dataeval.quality import Duplicates, Outliers
    from dataeval.scope import Prioritize
    from dataeval.shift import DriftKNeighbors, DriftUnivariate, DriftWasserstein, OODKNeighbors
    from verification.helpers import make_metadata

    images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
    embeddings = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    return [
        ("diversity-method", lambda: Diversity(method="bogus").evaluate(make_metadata()), "method"),
        ("outliers-threshold", lambda: Outliers(outlier_threshold="bogus").evaluate(images), "threshold"),
        ("duplicates-flags", lambda: Duplicates(flags=ImageStats.NONE).evaluate(images), "flags"),
        ("drift-univariate-method", lambda: DriftUnivariate(method="bogus"), "method"),
        ("drift-univariate-correction", lambda: DriftUnivariate(correction="bogus"), "correction"),
        ("drift-kneighbors-k", lambda: DriftKNeighbors(k=100).fit(embeddings), "k"),
        ("ood-kneighbors-k", lambda: OODKNeighbors(k=100).fit(embeddings), "k"),
        ("drift-wasserstein-ratio", lambda: DriftWasserstein(ratio_threshold=-1), "ratio_threshold"),
        ("prioritize-method", lambda: Prioritize(method="bogus").evaluate(embeddings), "method"),
        ("sufficiency-runs", lambda: Sufficiency(object(), runs=0), "runs"),
        ("sufficiency-substeps", lambda: Sufficiency(object(), substeps=0), "substeps"),
        ("embeddings-batch-size", lambda: Embeddings(images, batch_size=0), "batch_size"),
        ("bovw-vocab-size", lambda: BoVWExtractor(vocab_size=0), "vocab_size"),
        ("resize-size", lambda: Resize(-1), "size"),
        ("resize-mode", lambda: Resize(8, mode="bogus"), "mode"),
        ("select-channels", lambda: SelectChannels("bogus"), "channels"),
        ("crop-region", lambda: Crop((5, 5, 1, 1)), "region"),
        ("metadata-length-mismatch", lambda: Metadata.from_factors({"a": np.arange(3)}, np.arange(4)), "length"),
    ]


class TestInputContracts:
    """Verify input validation across key public API entry points."""

    def test_label_stats_accepts_valid_labels(self):
        from dataeval.core import label_stats

        result = label_stats(np.array([0, 1, 2, 0, 1, 2]))
        assert result is not None

    def test_label_stats_handles_empty_labels(self):
        """Empty labels should either raise or return an empty/zero-count result."""
        from dataeval.core import label_stats

        result = label_stats(np.array([], dtype=np.intp))
        # API handles empty input gracefully — verify result is consistent
        assert "label_count" in result
        assert result["label_count"] == 0

    def test_duplicates_handles_empty_dataset(self):
        """Empty dataset should either raise or return an empty result."""
        import polars as pl

        from dataeval.quality import Duplicates

        result = Duplicates().evaluate(np.array([]))
        assert isinstance(result.data(), pl.DataFrame)

    def test_outliers_handles_empty_dataset(self):
        """Empty dataset should either raise or return an empty result."""
        import polars as pl

        from dataeval.quality import Outliers

        result = Outliers().evaluate(np.array([]))
        assert isinstance(result.data(), pl.DataFrame)

    @pytest.mark.parametrize("case_id", INVALID_ARGUMENT_IDS)
    def test_invalid_argument_raises_error_naming_it(self, case_id: str):
        """Out-of-range, unknown, or mismatched arguments raise ValueError/TypeError that names the argument."""
        cases = {cid: (fn, name) for cid, fn, name in _invalid_argument_cases()}
        assert list(cases) == INVALID_ARGUMENT_IDS
        call, argument = cases[case_id]
        with pytest.raises((ValueError, TypeError), match=rf"\b{argument}\b"):
            call()


def _unvalidated_argument_cases():
    """Arguments that v1.1.4 accepts silently or rejects without naming them (see NFR-6)."""
    from dataeval.bias import Balance
    from dataeval.data import ClassBalance, Limit
    from dataeval.performance.schedules import GeometricSchedule, ManualSchedule
    from dataeval.quality import Duplicates
    from dataeval.scope import Prioritize
    from dataeval.shift import DriftMMD, DriftUnivariate, OODKNeighbors
    from verification.helpers import make_metadata

    return [
        ("balance-num-neighbors", lambda: Balance(num_neighbors=-1).evaluate(make_metadata()), "num_neighbors"),
        ("drift-univariate-p-val", lambda: DriftUnivariate(p_val=2.0), "p_val"),
        ("drift-mmd-n-permutations", lambda: DriftMMD(n_permutations=0), "n_permutations"),
        ("ood-kneighbors-threshold-perc", lambda: OODKNeighbors(threshold_perc=150), "threshold_perc"),
        ("prioritize-order", lambda: Prioritize(order="bogus"), "order"),
        ("geometric-substeps", lambda: GeometricSchedule(0), "substeps"),
        ("manual-negative-step", lambda: ManualSchedule([-1]), "eval_points"),
        ("limit-negative", lambda: Limit(-1), "size"),
        ("class-balance-method", lambda: ClassBalance("bogus"), "method"),
        ("duplicates-cluster-algorithm", lambda: Duplicates(cluster_algorithm="bogus"), "cluster_algorithm"),
    ]


# main validates these six since v1.1.4; the other four are still gaps.
_NOW_VALIDATED = (0, 1, 2, 3, 4, 9)
_STILL_UNVALIDATED = (5, 6, 7, 8)


@pytest.mark.parametrize("index", _NOW_VALIDATED)
def test_previously_unvalidated_arguments_are_now_rejected(index: int):
    """These were v1.1.4 gaps (see NFR-6); the argument is now rejected with an error that names it."""
    call, argument = _unvalidated_argument_cases()[index][1:]
    with pytest.raises((ValueError, TypeError), match=rf"\b{argument}\b"):
        call()


@pytest.mark.parametrize("index", _STILL_UNVALIDATED)
@pytest.mark.xfail(strict=True, reason="known gap: argument not validated, or error does not name it")
def test_known_unvalidated_arguments(index: int):
    """Not referenced by the registry. Strict xfail: becomes a failure (XPASS) once validation is added."""
    call, argument = _unvalidated_argument_cases()[index][1:]
    with pytest.raises((ValueError, TypeError), match=rf"\b{argument}\b"):
        call()
