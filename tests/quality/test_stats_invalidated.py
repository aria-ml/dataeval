"""Tests for the soft warning raised when a view operation invalidates requested stats."""

import warnings

import numpy as np
import pytest

from dataeval.core import compute_stats
from dataeval.data import View
from dataeval.data._view import Operation
from dataeval.exceptions import StatsInvalidatedWarning
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers
from dataeval.quality._shared import checked_compute_stats


class Resize(Operation):
    """Stand-in for the resize operation this machinery exists to cover."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    @property
    def invalidates(self) -> ImageStats:
        return ImageStats.DIMENSION | ImageStats.VISUAL_SHARPNESS | ImageStats.VISUAL_BRIGHTNESS

    def apply(self, view: View) -> None:
        pass


class Images:
    """Minimal image-only dataset with distinguishable images."""

    metadata = {"id": "images"}

    def __init__(self, n: int = 6) -> None:
        self._images = [np.full((3, 8, 8), i / 10.0, dtype=np.float32) for i in range(n)]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int):
        return self._images[index]


def resized_view(n: int = 6) -> View:
    return View(Images(n), [Resize((4, 4))])


def caught(fn):
    """Run ``fn`` and return the StatsInvalidatedWarnings it raised."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fn()
    return [w for w in record if issubclass(w.category, StatsInvalidatedWarning)]


@pytest.mark.required
class TestStatsInvalidatedWarning:
    def test_is_a_user_warning_not_a_future_warning(self):
        # This is a statement about data semantics, not about API lifecycle.
        assert issubclass(StatsInvalidatedWarning, UserWarning)
        assert not issubclass(StatsInvalidatedWarning, FutureWarning)


@pytest.mark.required
class TestCheckedComputeStats:
    def test_returns_one_result_per_dataset(self):
        results = checked_compute_stats(
            [Images(3), Images(4)], stats=ImageStats.DIMENSION_WIDTH, caller="Outliers", per_target=False
        )
        assert len(results) == 2
        assert len(results[0]["source_index"]) == 3
        assert len(results[1]["source_index"]) == 4

    def test_plain_dataset_does_not_warn(self):
        assert not caught(
            lambda: checked_compute_stats([Images(3)], stats=ImageStats.DIMENSION, caller="Outliers", per_target=False)
        )

    def test_warns_when_requested_stats_overlap_the_invalidation(self):
        assert len(caught(lambda: self._run(ImageStats.DIMENSION_WIDTH))) == 1

    def test_does_not_warn_when_requested_stats_do_not_overlap(self):
        # The op invalidates dimension/visual stats only; hashes are unaffected.
        assert not caught(lambda: self._run(ImageStats.HASH))

    def test_warns_once_for_n_datasets_sharing_operations(self):
        views = [resized_view(3), resized_view(3), resized_view(3)]
        found = caught(
            lambda: checked_compute_stats(views, stats=ImageStats.DIMENSION, caller="Outliers", per_target=False)
        )
        assert len(found) == 1

    def test_message_names_the_op_the_caller_the_stats_and_the_fix(self):
        found = caught(lambda: self._run(ImageStats.DIMENSION_WIDTH | ImageStats.VISUAL_SHARPNESS))
        message = str(found[0].message)
        assert "Resize(size=(4, 4))" in message
        assert "Outliers" in message
        assert "width" in message
        assert "sharpness" in message
        assert "transforms=" in message
        assert "flags=" in message

    def test_message_lists_only_the_intersected_stats(self):
        found = caught(lambda: self._run(ImageStats.DIMENSION_WIDTH))
        message = str(found[0].message)
        assert "width" in message
        assert "sharpness" not in message

    @staticmethod
    def _run(stats: ImageStats):
        return checked_compute_stats([resized_view()], stats=stats, caller="Outliers", per_target=False)


@pytest.mark.required
class TestEvaluatorIntegration:
    def test_outliers_warns_on_an_invalidating_view(self):
        found = caught(lambda: Outliers(flags=ImageStats.DIMENSION).evaluate(resized_view()))
        assert len(found) == 1
        assert "Outliers" in str(found[0].message)

    def test_outliers_warns_once_across_multiple_datasets(self):
        found = caught(lambda: Outliers(flags=ImageStats.DIMENSION).evaluate(resized_view(3), resized_view(3)))
        assert len(found) == 1

    def test_outliers_with_hash_flags_does_not_warn(self):
        # The op invalidates dimension/visual stats; hashes survive a resize's framing.
        assert not caught(lambda: Outliers(flags=ImageStats.HASH).evaluate(resized_view()))

    def test_outliers_does_not_warn_on_a_plain_dataset(self):
        assert not caught(lambda: Outliers(flags=ImageStats.DIMENSION).evaluate(Images()))

    def test_duplicates_with_all_flags_does_not_warn_about_dimension_stats(self):
        # Duplicates only ever computes `flags & ImageStats.HASH`; intersecting against
        # the requested flags rather than the effective ones would warn spuriously here.
        assert not caught(lambda: Duplicates(flags=ImageStats.ALL).evaluate(resized_view()))

    def test_duplicates_multi_with_all_flags_does_not_warn(self):
        assert not caught(lambda: Duplicates(flags=ImageStats.ALL).evaluate(resized_view(3), resized_view(3)))

    def test_from_stats_is_unchanged_and_unwarned(self):
        stats = compute_stats(resized_view(), stats=ImageStats.DIMENSION, per_target=False)
        assert not caught(lambda: Outliers(flags=ImageStats.DIMENSION).from_stats(stats))
