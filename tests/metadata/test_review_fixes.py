"""Regression tests for the level-restructure review findings.

Each test here pins down behavior that was wrong, ambiguous or silently
inconsistent when the level model landed.
"""

import logging

import numpy as np
import polars as pl
import pytest

from dataeval._metadata import Metadata
from dataeval._structurers import (
    RESERVED_COLUMNS,
    ICStructurer,
    ODImageStructurer,
    reserved_block_columns,
    select_structurer,
)
from dataeval.types._factors import FactorLevelSchema
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


def _od_target(count: int) -> ObjectDetectionTarget:
    return ObjectDetectionTarget(
        np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
        np.arange(count) % 2,
        np.full(count, 0.5),
    )


def _od_dataset(counts=(2, 3, 1), metadata=None) -> MockDataset:
    """Object detection dataset with a categorical and a numeric image factor."""
    return MockDataset(
        np.zeros((len(counts), 3, 16, 16)),
        [_od_target(count) for count in counts],
        metadata or [{"weather": "sun" if i % 2 else "rain", "hour": float(i)} for i in range(len(counts))],
    )


@pytest.mark.required
class TestFilterByRowsAndTypes:
    """filter_by_* returns target-level rows, and survives string factors."""

    def test_rows_align_with_class_labels(self):
        md = Metadata(_od_dataset())
        assert md.level_counts == {"unit": 3, "instance": 6}

        filtered = md.filter_by_factor(lambda *_: True)
        assert filtered.shape[0] == len(md.class_labels) == 6
        assert filtered.shape == md.factor_data.shape

    def test_categorical_string_factor_does_not_raise(self):
        """The raw column holds strings; the digitized companion is what casts."""
        md = Metadata(_od_dataset())
        categorical = md.filter_by_factor_type("categorical")
        assert categorical.shape == (6, 1)
        # Two distinct weather values across three images, propagated to their instances.
        assert set(np.unique(categorical).tolist()) == {0.0, 1.0}

    def test_continuous_factors_keep_their_raw_values(self):
        """feature_distance needs real values, not bin indices."""
        md = Metadata(_od_dataset(counts=(2, 2, 2)), continuous_factor_bins={"hour": 3})
        continuous = md.filter_by_factor_type("continuous")
        assert continuous.shape == (6, 1)
        assert continuous[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    def test_level_predicate_selects_image_factors_from_instance_rows(self):
        md = Metadata(_od_dataset())
        md.add_factors({"iou": np.arange(6, dtype=np.float64)}, level="instance")

        image_level = md.filter_by_factor(lambda _, fi: fi.level == "unit")
        instance_level = md.filter_by_factor(lambda _, fi: fi.level == "instance")
        assert image_level.shape == (6, 2)  # hour + weather, read from the instance rows
        assert instance_level.shape == (6, 1)  # iou
        assert instance_level[:, 0].tolist() == list(range(6))

    def test_empty_selection_is_empty(self):
        md = Metadata(_od_dataset())
        assert md.filter_by_factor(lambda *_: False).size == 0


@pytest.mark.required
class TestRenameWarningSite:
    """The FactorInfo.level warning fires for handouts, not for internal reads."""

    def test_factor_data_does_not_warn(self, recwarn):
        md = Metadata(_od_dataset())
        _ = md.factor_data
        assert not [w for w in recwarn.list if "FactorInfo.level" in str(w.message)]

    def test_warning_survives_an_internal_read(self):
        """The once-per-instance budget is not spent by factor_data."""
        md = Metadata(_od_dataset())
        _ = md.factor_data
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            _ = md.factor_info

    def test_is_discrete_does_not_warn(self, recwarn):
        md = Metadata(_od_dataset())
        _ = md.is_discrete
        assert not [w for w in recwarn.list if "FactorInfo.level" in str(w.message)]


@pytest.mark.required
class TestAmbiguousAutoLevel:
    """Levels of equal size must not break add_factors(level="auto")."""

    def test_one_detection_per_image_adds_without_warning(self, recwarn):
        """Equal row counts that correspond one-to-one are not ambiguous."""
        md = Metadata(_od_dataset(counts=(1, 1, 1)))
        assert md.level_counts == {"unit": 3, "instance": 3}

        md.add_factors({"foo": np.arange(3)})

        # The coarsest match wins, as it always has. Either level would put the same
        # values on the target rows here, so there is nothing to warn about.
        assert md.factor_info["foo"].level == "unit"
        assert md.rows_at("unit")["foo"].to_list() == [0, 1, 2]
        assert not [w for w in recwarn.list if "matches the" in str(w.message)]

    def test_fully_labelled_classification_adds_without_warning(self, recwarn):
        """The common case: image count == instance count on every labelled IC dataset."""
        md = Metadata(MockDataset(np.zeros((3, 3, 4, 4)), np.eye(3), [{"a": i} for i in range(3)]))
        assert md.level_counts == {"unit": 3, "instance": 3}

        md.add_factors({"bright": np.arange(3.0)})
        assert md.factor_info["bright"].level == "unit"
        assert not [w for w in recwarn.list if "matches the" in str(w.message)]

    def test_equal_counts_that_do_not_correspond_do_warn(self):
        """3 images and 3 detections, but spread 0/1/2 — the choice changes the data."""
        md = Metadata(_od_dataset(counts=(0, 1, 2)))
        assert md.level_counts == {"unit": 3, "instance": 3}

        with pytest.warns(UserWarning, match="do not correspond one-to-one"):
            md.add_factors({"foo": np.arange(3)})
        assert md.factor_info["foo"].level == "unit"

    def test_unambiguous_length_does_not_warn(self, recwarn):
        md = Metadata(_od_dataset())
        md.add_factors({"iou": np.arange(6, dtype=np.float64)})
        assert md.factor_info["iou"].level == "instance"
        assert not [w for w in recwarn.list if "matches the" in str(w.message)]

    def test_length_matching_no_level_still_raises(self):
        from dataeval.exceptions import ShapeMismatchError

        md = Metadata(_od_dataset())
        with pytest.raises(ShapeMismatchError, match="different length"):
            md.add_factors({"foo": np.arange(99)})


@pytest.mark.required
class TestArrayProtocolConsistency:
    """len, shape and iteration describe the same rows."""

    def test_len_matches_shape_for_object_detection(self):
        md = Metadata(_od_dataset())
        assert len(md) == md.shape[0] == 6
        assert len(list(md)) == len(md)
        assert md.item_count == 3

    def test_len_matches_shape_for_classification(self):
        md = Metadata(MockDataset(np.zeros((4, 3, 4, 4)), np.eye(4, 2)[[0, 1, 0, 1]]))
        assert len(md) == md.shape[0] == 4
        assert md.item_count == 4


@pytest.mark.required
class TestBindResetsLevelState:
    def test_rebinding_clears_the_previous_schema(self):
        md = Metadata(_od_dataset())
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            _ = md.factor_info
        assert md.level_counts == {"unit": 3, "instance": 6}

        md.bind(MockDataset(np.zeros((2, 3, 4, 4)), np.eye(2)))
        # Read before any structuring, exactly as a stale attribute read would: the
        # level state must be back to its defaults, not the previous dataset's.
        assert md._label_level == "unit"
        assert md._view_level == "unit"
        assert list(md._levels) == ["unit"]
        assert md._layout.counts == {}
        assert md._factors_by_level == {}

        # And structuring then answers for the dataset that is actually bound.
        assert md.level_counts == {"unit": 2, "instance": 2}

    def test_rebinding_restores_the_rename_warning(self):
        md = Metadata(_od_dataset())
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            _ = md.factor_info

        md.bind(_od_dataset())
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            _ = md.factor_info


@pytest.mark.required
class TestBlockColumnValidation:
    def test_ragged_column_is_rejected_at_its_source(self):
        with pytest.raises(ValueError, match="must have 3 values"):
            reserved_block_columns("instance", 3, class_label=[0, 1], score=[0.5, 0.5, 0.5])

    def test_frame_index_is_not_reserved(self):
        """No structurer writes it, so it must not cost a user their metadata key."""
        assert "frame_index" not in RESERVED_COLUMNS

        md = Metadata(
            MockDataset(
                np.zeros((2, 3, 4, 4)),
                np.eye(2),
                [{"frame_index": i} for i in range(2)],
            ),
        )
        assert "frame_index" in md.dataframe.columns
        assert "metadata_frame_index" not in md.dataframe.columns


@pytest.mark.required
class TestLevelSchemaDuplicates:
    def test_of_rejects_a_repeated_level(self):
        with pytest.raises(ValueError, match="appear more than once"):
            FactorLevelSchema.of("unit", "unit")


@pytest.mark.required
class TestFirstDatumIsReadOnce:
    """The task probe hands its datum to the structurer instead of re-reading it."""

    class CountingDataset(MockDataset):
        def __init__(self, data, targets, metadata=None):
            super().__init__(data, targets, metadata)
            self.reads: list[int] = []

        def __getitem__(self, idx):
            self.reads.append(idx)
            return super().__getitem__(idx)

    @staticmethod
    def _reads_during_structuring(dataset, **kwargs) -> list[int]:
        """Reads made by structuring alone, excluding construction-time validation."""
        md = Metadata(dataset, **kwargs)
        dataset.reads.clear()
        md._structure()
        return dataset.reads

    def test_object_detection_reads_each_item_once(self):
        dataset = self.CountingDataset(np.zeros((3, 3, 4, 4)), [_od_target(2) for _ in range(3)])
        assert self._reads_during_structuring(dataset) == [0, 1, 2]

    def test_classification_reads_each_item_once(self):
        dataset = self.CountingDataset(np.zeros((3, 3, 4, 4)), np.eye(3))
        assert self._reads_during_structuring(dataset) == [0, 1, 2]

    def test_explicit_task_reads_each_item_once(self):
        dataset = self.CountingDataset(np.zeros((3, 3, 4, 4)), np.eye(3))
        assert self._reads_during_structuring(dataset, task="IC") == [0, 1, 2]

    def test_a_structurer_built_directly_still_reads_everything(self):
        dataset = self.CountingDataset(np.zeros((2, 3, 4, 4)), np.eye(2))
        ICStructurer().build(dataset)
        assert dataset.reads == [0, 1]

    def test_selected_structurer_matches_the_task(self):
        dataset = self.CountingDataset(np.zeros((2, 3, 4, 4)), [_od_target(1) for _ in range(2)])
        assert isinstance(select_structurer(dataset), ODImageStructurer)


@pytest.mark.required
class TestUnlabeledClassificationItems:
    """An unlabeled IC item keeps its item row and every factor on it.

    Before the item and target levels were separated, such an item vanished from the
    dataframe entirely — taking its metadata with it — because image classification
    had a single level serving as both.
    """

    @staticmethod
    def _partially_labeled() -> MockDataset:
        """3 images, item 1 carries an empty target array."""
        return MockDataset(
            np.zeros((3, 3, 4, 4)),
            [np.eye(2)[0], np.array([]), np.eye(2)[1]],
            [{"weather": w, "brightness": b} for w, b in (("sun", 0.1), ("fog", 0.9), ("rain", 0.5))],
        )

    def test_item_row_and_factors_survive(self):
        md = Metadata(self._partially_labeled())

        assert md.level_counts == {"unit": 3, "instance": 2}
        assert md.item_count == 3
        assert md.rows_at("unit").height == 3
        # The unlabeled item's metadata is still here — this is the whole point.
        assert md.rows_at("unit")["weather"].to_list() == ["sun", "fog", "rain"]
        assert md.rows_at("unit")["brightness"].to_list() == [0.1, 0.9, 0.5]

    def test_label_aware_views_cover_only_labelled_items(self):
        md = Metadata(self._partially_labeled())

        assert md.rows_at(md.label_level).height == 2
        assert md.class_labels.tolist() == [0, 1]
        assert md.item_indices.tolist() == [0, 2]
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_unlabeled_items_are_named_in_the_log(self, caplog):
        with caplog.at_level(logging.INFO, logger="dataeval.metadata"):
            Metadata(self._partially_labeled())._structure()

        assert "[1]" in caplog.text
        assert "carried no target" in caplog.text

    def test_per_image_stats_can_be_added_by_source_index(self):
        """The workflow the collapsed level model made impossible."""
        from dataeval.types import SourceIndex

        md = Metadata(self._partially_labeled())
        # compute_stats labels one entry per dataset item, including the unlabeled one.
        source_index = [SourceIndex(i, None, None) for i in range(3)]
        md.add_factors({"sharpness": np.array([10.0, 20.0, 30.0])}, source_index=source_index)

        assert md.factor_info["sharpness"].level == "unit"
        assert md.rows_at("unit")["sharpness"].to_list() == [10.0, 20.0, 30.0]
        # Image-level values reach the target rows of the items that have them.
        assert md.rows_at(md.label_level)["sharpness"].to_list() == [10.0, 30.0]

    def test_a_genuinely_wrong_length_still_raises(self):
        from dataeval.exceptions import ShapeMismatchError
        from dataeval.types import SourceIndex

        md = Metadata(self._partially_labeled())
        source_index = [SourceIndex(i, None, None) for i in range(2)]
        with pytest.raises(ShapeMismatchError, match="item_indices"):
            md.add_factors({"sharpness": np.arange(2.0)}, source_index=source_index)


@pytest.mark.required
class TestPaddedColumnDtype:
    """The same factor gets the same dtype whether or not padding was needed."""

    def test_binned_column_dtype_matches_across_tasks(self):
        values = [0.0, 1.0, 2.0, 3.0]
        ic = Metadata(
            MockDataset(
                np.zeros((4, 3, 4, 4)),
                np.eye(4, 2)[[0, 1, 0, 1]],
                [{"hour": value} for value in values],
            ),
            continuous_factor_bins={"hour": 2},
        )
        od = Metadata(
            _od_dataset(counts=(1, 1, 1, 1), metadata=[{"hour": value} for value in values]),
            continuous_factor_bins={"hour": 2},
        )
        _, _ = ic.factor_data, od.factor_data

        binned = "hour↕"
        assert ic.dataframe.schema[binned] == od.dataframe.schema[binned] == pl.Int64


@pytest.mark.required
class TestLevelModelIsExposed:
    """Task-generic code must not have to hardcode level names or reach into privates."""

    def test_item_and_label_levels_are_public(self):
        od = Metadata(_od_dataset())
        assert (od.item_level, od.label_level) == ("unit", "instance")
        assert od.rows_at(od.item_level).height == 3
        assert od.rows_at(od.label_level).height == 6

    def test_multi_target_distinguishes_the_tasks(self):
        """Neither row counts nor level names can: both tasks name their levels the same,
        and an OD dataset with one detection per image has equal counts."""
        one_each = Metadata(_od_dataset(counts=(1, 1, 1)))
        assert one_each.level_counts["unit"] == one_each.level_counts["instance"] == 3
        assert one_each.multi_target is True

        ic = Metadata(MockDataset(np.zeros((3, 3, 4, 4)), np.eye(3)))
        assert ic.item_level == "unit"
        assert ic.label_level == "instance"
        assert ic.multi_target is False

    def test_filter_by_factor_can_name_the_level(self):
        """The predicate in the docstring, without hardcoding "instance"."""
        md = Metadata(_od_dataset())
        md.add_factors({"iou": np.arange(6.0)}, level="instance")

        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            native = md.filter_by_factor(lambda _, fi: fi.level == md.label_level)
        assert native.shape == (6, 1)

    def test_an_empty_selection_still_has_the_view_row_count(self):
        md = Metadata(_od_dataset())
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            none_native = md.filter_by_factor(lambda _, fi: fi.level == md.label_level)
        assert none_native.shape == (6, 0)


@pytest.mark.required
class TestView:
    """The view chooses which rows the array accessors project."""

    def test_default_view_is_the_label_level(self):
        md = Metadata(_od_dataset())
        assert md.view == md.label_level == "instance"
        assert md.factor_data.shape[0] == len(md.class_labels) == 6

    def test_at_reads_a_factor_once_per_entity(self):
        """The whole point: an image factor read from instance rows is weighted by
        detections-per-image, and read from image rows is not."""
        md = Metadata(_od_dataset(counts=(3, 1, 1)))

        zoomed = md.at("unit")
        instance_view = md.factor_data
        image_view = zoomed.factor_data

        assert instance_view.shape[0] == 5
        assert image_view.shape[0] == 3

        # Same factor, same bin ids — only how many rows carry each one changes. The
        # first image has three detections, so the instance view triples its value.
        column = list(md.factor_names).index("weather")
        assert list(md.factor_names) == list(zoomed.factor_names)
        assert image_view[:, column].tolist() == [0, 1, 0]
        assert instance_view[:, column].tolist() == [0, 0, 0, 1, 0]

    def test_at_leaves_the_original_alone(self):
        md = Metadata(_od_dataset())
        zoomed = md.at("unit")

        assert zoomed.view == "unit"
        assert md.view == "instance"
        assert md.factor_data.shape[0] == 6

    def test_at_copies_are_independent(self):
        md = Metadata(_od_dataset())
        zoomed = md.at("unit")
        zoomed.add_factors({"extra": np.arange(3.0)}, level="unit")

        assert "extra" in zoomed.factor_names
        assert "extra" not in md.factor_names

    def test_setting_the_view_rebuilds_in_place(self):
        md = Metadata(_od_dataset())
        assert md.factor_data.shape[0] == 6

        md.view = "unit"
        assert md.factor_data.shape[0] == 3

        md.view = "instance"
        assert md.factor_data.shape[0] == 6

    def test_an_unknown_view_raises(self):
        md = Metadata(_od_dataset())
        with pytest.raises(ValueError, match="Unknown level"):
            md.view = "sequence"  # type: ignore[assignment]

    def test_class_labels_refuses_a_view_above_the_label_level(self):
        """Silently returning instance labels for image rows would misalign every evaluator."""
        md = Metadata(_od_dataset()).at("unit")
        with pytest.raises(ValueError, match="no label per row"):
            _ = md.class_labels

    def test_constructor_view_is_validated_against_the_schema(self):
        md = Metadata(_od_dataset(), view="sequence")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown level"):
            _ = md.factor_data

    def test_bind_clears_an_explicit_view(self):
        md = Metadata(_od_dataset(), view="unit")
        assert md.view == "unit"

        md.bind(_od_dataset())
        assert md.view == "instance"


@pytest.mark.required
class TestEmptyFactorProjection:
    """len(), shape and iteration have to describe the same rows, factors or not."""

    def _no_factors(self) -> Metadata:
        return Metadata(_od_dataset(metadata=[{}, {}, {}]))

    def test_shape_len_and_iteration_agree(self):
        md = self._no_factors()

        assert not md.factor_names
        assert md.shape == (6, 0)
        assert len(md) == 6
        assert len(list(md)) == 6
        assert md[0].shape == (0,)


@pytest.mark.required
class TestFirstDatumIsReleased:
    """A structurer outlives build(), so it must not pin the datum it was handed."""

    def test_metadata_does_not_retain_a_decoded_item(self):
        md = Metadata(_od_dataset())
        _ = md.dataframe

        assert md._structurer is not None
        assert getattr(md._structurer, "_first_datum", None) is None


@pytest.mark.required
class TestEmptyDatasetWarning:
    """The empty-dataset fallback warns from the frame that knows the caller."""

    def test_warning_points_at_the_construction_site(self):
        with pytest.warns(UserWarning, match="Cannot infer a task from an empty dataset") as record:
            Metadata(MockDataset([], []))

        assert record[0].filename == __file__

    def test_an_explicit_task_silences_it(self, recwarn):
        Metadata(MockDataset([], []), task="IC")
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_bind_warns_too(self):
        md = Metadata()
        with pytest.warns(UserWarning, match="Cannot infer a task from an empty dataset") as record:
            md.bind(MockDataset([], []))

        assert record[0].filename == __file__


@pytest.mark.required
class TestBinsSurviveTheView:
    """A factor's bins belong to the factor, not to whatever view was current."""

    def test_at_bins_factors_the_source_view_never_saw(self):
        """Moving down exposes finer factors; they must not be counted but unprocessed."""
        md = Metadata(_od_dataset(), view="unit")
        md.add_factors({"iou": np.arange(6.0)}, level="instance")
        md.factor_data  # noqa: B018  - bins the image factors at the image view

        zoomed = md.at("instance")
        assert list(zoomed.factor_names) == ["hour", "iou", "weather"]
        # The three views of "how many factors are there" have to agree.
        assert zoomed.shape == (6, 3)
        assert zoomed.factor_data.shape == (6, 3)
        assert len(zoomed.is_discrete) == 3
        assert set(zoomed.factor_info) == {"hour", "iou", "weather"}

    def test_round_trip_restores_info_without_rebinning(self):
        """A factor that leaves the visible set and returns keeps the column's meaning."""
        md = Metadata(_od_dataset())
        md.add_factors({"iou": np.arange(6.0)}, level="instance")
        original = dict(md.factor_info)

        # 'iou' is invisible from the image view, then visible again from instance.
        assert "iou" not in md.at("unit").factor_names
        assert md.at("unit").at("instance").factor_info == original

    def test_view_assignment_keeps_bins(self):
        md = Metadata(_od_dataset())
        md.factor_data  # noqa: B018
        binned = [c for c in md.dataframe.columns if c.endswith(("#", "%"))]

        md.view = "unit"
        md.view = "instance"
        assert [c for c in md.dataframe.columns if c.endswith(("#", "%"))] == binned
        assert set(md.factor_info) == {"hour", "weather"}


@pytest.mark.required
class TestResetBinsIsNotGuarded:
    """_reset_bins must drop companion columns whatever _is_binned currently says."""

    def test_overwrite_after_exclude_rebins(self):
        """exclude clears _is_binned but leaves the columns; the overwrite must still land."""
        md = Metadata(_od_dataset())
        md.factor_data  # noqa: B018
        md.exclude = ["weather"]

        md.add_factors({"hour": np.array([100.0, 200.0, 300.0])}, level="unit", overwrite=True)

        assert list(md.factor_names) == ["hour"]
        assert md.shape == (6, 1)
        assert md.factor_data.shape == (6, 1)
        assert "hour" in md.factor_info

    def test_continuous_factor_bins_after_exclude_takes_effect(self):
        md = Metadata(_od_dataset(counts=(1, 1, 1)))
        md.factor_data  # noqa: B018
        md.exclude = ["weather"]

        md.continuous_factor_bins = {"hour": 2}
        assert len(np.unique(md.factor_data[:, 0])) <= 2


@pytest.mark.required
class TestViewAwareAccessors:
    def test_item_indices_matches_the_view_it_is_read_at(self):
        md = Metadata(_od_dataset())
        assert md.item_indices.tolist() == [0, 0, 1, 1, 1, 2]

        image = md.at("unit")
        assert len(image.item_indices) == image.factor_data.shape[0] == 3
        assert image.item_indices.tolist() == [0, 1, 2]

    def test_at_announces_the_level_rename_on_its_own(self):
        """A copy is a fresh object in the user's hands and gets its own warning budget."""
        md = Metadata(_od_dataset())
        with pytest.warns(DeprecationWarning, match="FactorInfo.level"):
            md.factor_info  # noqa: B018
        with pytest.warns(DeprecationWarning, match="FactorInfo.level"):
            md.at("unit").factor_info  # noqa: B018


@pytest.mark.required
class TestInstanceLevelKeyColumn:
    """Every structurer that declares the instance level writes its key column."""

    @pytest.mark.parametrize("dataset", ["ic", "od"])
    def test_instance_index_is_present_for_both_tasks(self, dataset):
        md = Metadata(
            _od_dataset()
            if dataset == "od"
            else MockDataset(np.zeros((3, 3, 16, 16)), np.eye(3)[[0, 1, 0]], [{"weather": "sun"}] * 3),
        )
        assert "instance_index" in md.rows_at("instance").columns

    def test_od_logs_images_without_detections(self, caplog):
        with caplog.at_level(logging.INFO, logger="dataeval.metadata"):
            Metadata(_od_dataset(counts=(2, 0, 1)))._structure()

        assert "carried no target" in caplog.text
        assert "'instance' rows" in caplog.text


@pytest.mark.required
class TestAddFactorsDtype:
    """add_factors anchors the column dtype on the native values, as _add_level_column does."""

    def test_empty_level_keeps_the_declared_dtype(self):
        md = Metadata(_od_dataset(counts=(0, 0, 0)))
        md.add_factors({"iou": np.array([], dtype=np.float64)}, level="instance")
        assert md.dataframe.schema["iou"] == pl.Float64

    def test_numeric_width_is_preserved(self):
        md = Metadata(_od_dataset())
        md.add_factors({"small": np.arange(6, dtype=np.int8)}, level="instance")
        assert md.dataframe.schema["small"] == pl.Int8

    def test_string_factor_is_still_a_string(self):
        md = Metadata(_od_dataset())
        md.add_factors({"tag": np.array(["a", "b", "c"])}, level="unit")
        assert md.dataframe.schema["tag"] == pl.String
