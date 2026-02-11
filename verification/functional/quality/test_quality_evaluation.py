"""Verify that data quality evaluators produce correct output types.

Maps to meta repo test cases:
  - TC-3.1: Data quality analysis (Duplicates, Outliers)
"""

import numpy as np
import pytest


@pytest.mark.test_case("3-1")
class TestQualityEvaluation:
    """Verify Duplicates and Outliers evaluators."""

    def test_duplicates_detects_exact_copies(self):
        from dataeval.quality import Duplicates

        rng = np.random.default_rng(0)
        images = rng.random((10, 3, 16, 16)).astype(np.float32)
        # Add exact duplicates
        images_with_dupes = np.concatenate([images, images[:3]])
        result = Duplicates().evaluate(images_with_dupes)
        assert hasattr(result, "items")
        assert hasattr(result, "targets")

    def test_duplicates_items_has_exact_field(self):
        from dataeval.quality import Duplicates

        rng = np.random.default_rng(0)
        images = rng.random((10, 3, 16, 16)).astype(np.float32)
        images_with_dupes = np.concatenate([images, images[:3]])
        result = Duplicates().evaluate(images_with_dupes)
        assert hasattr(result.items, "exact")

    def test_outliers_returns_issues_dataframe(self):
        import polars as pl

        from dataeval.quality import Outliers

        rng = np.random.default_rng(0)
        images = rng.random((50, 3, 16, 16)).astype(np.float32)
        result = Outliers().evaluate(images)
        assert hasattr(result, "issues")
        assert isinstance(result.issues, pl.DataFrame)

    def test_outliers_supports_zscore_method(self):
        from dataeval.quality import Outliers

        rng = np.random.default_rng(0)
        images = rng.random((50, 3, 16, 16)).astype(np.float32)
        result = Outliers(outlier_method="zscore").evaluate(images)
        assert hasattr(result, "issues")

    def test_outliers_supports_iqr_method(self):
        from dataeval.quality import Outliers

        rng = np.random.default_rng(0)
        images = rng.random((50, 3, 16, 16)).astype(np.float32)
        result = Outliers(outlier_method="iqr").evaluate(images)
        assert hasattr(result, "issues")

    def test_quality_outputs_support_meta(self):
        from dataeval.quality import Duplicates, Outliers

        rng = np.random.default_rng(0)
        images = rng.random((20, 3, 16, 16)).astype(np.float32)

        dup_result = Duplicates().evaluate(images)
        assert dup_result.meta() is not None

        out_result = Outliers().evaluate(images)
        assert out_result.meta() is not None
