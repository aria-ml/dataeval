"""Verify that data quality evaluators produce correct output types.

Maps to meta repo test cases:
  - TC-3.1: Data quality analysis (Duplicates, Outliers)
"""

import numpy as np


class TestQualityEvaluation:
    """Verify Duplicates and Outliers evaluators."""

    def test_duplicates_detects_exact_copies(self):
        import polars as pl

        from dataeval.quality import Duplicates

        rng = np.random.default_rng(0)
        images = rng.random((10, 3, 16, 16)).astype(np.float32)
        # Add exact duplicates
        images_with_dupes = np.concatenate([images, images[:3]])
        result = Duplicates().evaluate(images_with_dupes)
        assert isinstance(result.data(), pl.DataFrame)

    def test_duplicates_items_has_exact_field(self):
        from dataeval.quality import Duplicates

        rng = np.random.default_rng(0)
        images = rng.random((10, 3, 16, 16)).astype(np.float32)
        images_with_dupes = np.concatenate([images, images[:3]])
        result = Duplicates().evaluate(images_with_dupes)
        df = result.data()
        exact_items = df.filter((df["level"] == "item") & (df["dup_type"] == "exact"))
        assert exact_items.shape[0] > 0

    def test_duplicates_detects_near_duplicates(self):
        """Verify Duplicates detects near duplicates using perceptual hashing."""
        import polars as pl

        from dataeval.quality import Duplicates

        rng = np.random.default_rng(0)
        data = rng.random((20, 3, 16, 16)).astype(np.float32)
        # Adding 0.001 creates non-byte-identical values detected as near duplicates via phash/dhash
        images_with_near_dupes = np.concatenate([data, data[:2] + 0.001])
        result = Duplicates().evaluate(images_with_near_dupes)
        df = result.data()
        assert isinstance(df, pl.DataFrame)
        near_items = df.filter((df["level"] == "item") & (df["dup_type"] == "near"))
        assert near_items.shape[0] > 0
        assert hasattr(result, "near")
        assert len(result.near) > 0

    def test_outliers_returns_issues_dataframe(self):
        import polars as pl

        from dataeval.quality import Outliers

        rng = np.random.default_rng(0)
        images = rng.random((50, 3, 16, 16)).astype(np.float32)
        result = Outliers().evaluate(images)
        assert isinstance(result.data(), pl.DataFrame)

    def test_outliers_supports_zscore_threshold(self):
        import polars as pl

        from dataeval.quality import Outliers
        from dataeval.utils.thresholds import ZScoreThreshold

        rng = np.random.default_rng(0)
        images = rng.random((50, 3, 16, 16)).astype(np.float32)
        result = Outliers(outlier_threshold=ZScoreThreshold()).evaluate(images)
        assert isinstance(result.data(), pl.DataFrame)

    def test_outliers_supports_iqr_threshold(self):
        import polars as pl

        from dataeval.quality import Outliers
        from dataeval.utils.thresholds import IQRThreshold

        rng = np.random.default_rng(0)
        images = rng.random((50, 3, 16, 16)).astype(np.float32)
        result = Outliers(outlier_threshold=IQRThreshold()).evaluate(images)
        assert isinstance(result.data(), pl.DataFrame)

    def test_quality_outputs_support_meta(self):
        from dataeval.quality import Duplicates, Outliers

        rng = np.random.default_rng(0)
        images = rng.random((20, 3, 16, 16)).astype(np.float32)

        dup_result = Duplicates().evaluate(images)
        assert dup_result.meta() is not None

        out_result = Outliers().evaluate(images)
        assert out_result.meta() is not None

    def test_duplicates_from_stats_across_datasets(self):
        """Duplicates.from_stats compares precomputed hashes from two datasets."""
        from dataeval.core import compute_stats
        from dataeval.flags import ImageStats
        from dataeval.quality import Duplicates

        rng = np.random.default_rng(0)
        first = rng.random((10, 3, 16, 16)).astype(np.float32)
        second = np.concatenate([rng.random((5, 3, 16, 16)).astype(np.float32), first[:3]])
        stats = [
            compute_stats(d, stats=ImageStats.HASH_DUPLICATES_BASIC, normalize_pixel_values=False)
            for d in (first, second)
        ]

        df = Duplicates().from_stats(stats).data()
        assert "dataset_indices" in df.columns
        exact = df.filter(df["dup_type"] == "exact")
        assert exact.shape[0] == 3
        for datasets in exact["dataset_indices"]:
            assert sorted(datasets) == [0, 1]  # each group spans both datasets
        assert sorted(sorted(items) for items in exact["item_indices"]) == [[0, 5], [1, 6], [2, 7]]

    def test_duplicates_and_outliers_on_object_detection_dataset(self):
        """Duplicates and Outliers report per detection target when asked to."""
        from dataeval.quality import Duplicates, Outliers
        from verification.helpers import SimpleODDataset

        rng = np.random.default_rng(0)
        images = (0.5 + 0.02 * rng.standard_normal((30, 3, 32, 32))).astype(np.float32)
        images[7] = 0.0  # uniform black image
        images[12] = images[3]  # exact copy
        dataset = SimpleODDataset(images, [np.array([0, 1])] * 30)

        dup = Duplicates().evaluate(dataset, per_target=True).data()
        targets = dup.filter(dup["level"] == "target")
        assert targets.shape[0] > 0
        assert sorted(targets["item_indices"][0]) == [3, 12]
        assert "target_indices" in dup.columns
        assert set(targets["target_indices"][0]) <= {0, 1}

        out = Outliers().evaluate(dataset, per_target=True).data()
        assert "target_index" in out.columns
        flagged = out.filter(out["target_index"].is_not_null())
        assert flagged.shape[0] > 0
        assert set(flagged["item_index"]) == {7}
        assert set(flagged["target_index"]) == {0, 1}
