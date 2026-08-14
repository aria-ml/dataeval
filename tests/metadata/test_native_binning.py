"""Factors are binned at their own level, not at the target level.

The behaviour these pin down: a factor's bin edges, its bin count and its
continuous/discrete verdict are read off the level where it holds one value per
entity. Binning at the target level instead reads each factor's distribution
through however many descendants an entity happens to have.
"""

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._metadata._columns import to_col
from dataeval.core._bin import bin_data
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


def _target(count: int) -> ObjectDetectionTarget:
    if not count:
        return ObjectDetectionTarget(np.empty((0, 4)), np.empty(0), np.empty(0))
    return ObjectDetectionTarget(
        np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
        np.arange(count) % 2,
        np.full(count, 0.5),
    )


def _od(counts, factors, **kwargs) -> Metadata:
    """Object detection over ``len(counts)`` images with the given detection counts."""
    dataset = MockDataset(
        np.zeros((len(counts), 3, 16, 16)),
        [_target(count) for count in counts],
        [{name: values[i] for name, values in factors.items()} for i in range(len(counts))],
    )
    return Metadata(dataset, **kwargs)


def _companion(md: Metadata, name: str) -> str:
    """The binned or digitized column backing a factor."""
    return to_col(name, md.factor_info[name])


@pytest.mark.required
class TestNativeLevelBinning:
    def test_binned_column_is_populated_at_the_factors_own_level(self):
        """An image-level factor carries its bin on image rows, not only on instances."""
        md = _od([2, 1, 2], {"brightness": [0.1, 0.5, 0.9]})
        column = _companion(md, "brightness")

        assert md.factor_info["brightness"].level == "unit"
        assert md.rows_at("unit")[column].to_list() == [0, 1, 2]

    def test_bin_assignment_is_the_same_from_either_level(self):
        """The invariant: a level projection must not change what a bin means."""
        md = _od([3, 1, 2, 4], {"brightness": [0.1, 0.5, 0.9, 0.3]})
        column = _companion(md, "brightness")

        at_image = md.rows_at("unit")[column].to_list()
        instances = md.target_data
        gathered = [instances.filter(instances["item_index"] == i)[column][0] for i in range(4)]

        assert at_image == gathered

    def test_item_without_targets_still_reaches_the_binner(self):
        """The regression: a childless item contributes no target row at all.

        Binning over target rows never saw its value, so it was absent from the
        edges and had no bin of its own.
        """
        md = _od([2, 1, 2, 0], {"brightness": [0.1, 0.5, 0.9, 99.0]})
        column = _companion(md, "brightness")

        assert md.rows_at("unit")["brightness"].to_list() == [0.1, 0.5, 0.9, 99.0]
        assert md.rows_at("unit")[column].to_list() == [0, 1, 2, 3]

    def test_edges_come_from_the_native_distribution(self):
        """uniform_count quantiles are density-weighted when taken over target rows."""
        rng = np.random.default_rng(0)
        counts = [1] * 50 + [10] * 50
        values = np.sort(rng.uniform(0.0, 1.0, len(counts)))

        md = _od(counts, {"b": values.tolist()}, auto_bin_method="uniform_count")
        column = _companion(md, "b")

        assert md.factor_info["b"].factor_type == "continuous"
        np.testing.assert_array_equal(md.rows_at("unit")[column].to_numpy(), bin_data(values, "uniform_count"))

    def test_target_level_factors_are_unchanged(self):
        """The compatibility pin: nothing moves for a factor already at the target level."""
        md = _od([2, 1, 2], {"brightness": [0.1, 0.5, 0.9]})
        md.add_factors({"iou": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}, level="instance")
        column = _companion(md, "iou")

        assert md.factor_info["iou"].level == "instance"
        assert md.target_data[column].to_list() == md.rows_at("instance")[column].to_list()


@pytest.mark.required
class TestImageClassificationUnaffected:
    """IC factors sit at image level over a fully labelled dataset, so nothing moves."""

    def _ic(self, labels, factors) -> Metadata:
        """``labels`` is a sequence of one-hot rows; an empty row means unlabelled."""
        return Metadata(
            MockDataset(
                np.zeros((len(labels), 3, 16, 16)),
                [np.asarray(row, dtype=float) for row in labels],
                [{name: values[i] for name, values in factors.items()} for i in range(len(labels))],
            ),
        )

    def test_factor_data_shape_and_alignment_hold(self):
        md = self._ic(np.eye(4, 2)[[0, 1, 0, 1]], {"brightness": [0.1, 0.5, 0.9, 0.3]})

        assert md.factor_data.shape == (4, 1)
        assert len(md.class_labels) == 4

    def test_unlabeled_image_keeps_its_factor_binned(self):
        """An unlabelled image has an image row but no instance row.

        Its factor value was previously invisible to the binner, which read only
        target rows; it now has a bin like every other image.
        """
        one_hot = np.eye(2)
        labels = [one_hot[0], one_hot[1], np.empty(0), one_hot[1]]  # image 2 carries no label
        md = self._ic(labels, {"brightness": [0.1, 0.5, 99.0, 0.3]})
        column = _companion(md, "brightness")

        assert md.level_counts["unit"] == 4
        assert md.level_counts["instance"] == 3
        assert None not in md.rows_at("unit")[column].to_list()
