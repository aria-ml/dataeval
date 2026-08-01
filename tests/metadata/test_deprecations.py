"""The compatibility surface left behind by the level restructure."""

import copy

import numpy as np
import pytest

from dataeval._metadata import Metadata
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


def _od_metadata() -> Metadata:
    """Object detection over 3 images with 2, 1 and 2 detections."""
    counts = (2, 1, 2)
    targets = [
        ObjectDetectionTarget(
            np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
            np.arange(count),
            np.full(count, 0.5),
        )
        for count in counts
    ]
    dataset = MockDataset(
        np.zeros((3, 3, 16, 16)),
        targets,
        [{"weather": value} for value in ("sun", "rain", "sun")],
    )
    return Metadata(dataset)


def _ic_metadata() -> Metadata:
    dataset = MockDataset(
        np.zeros((4, 3, 16, 16)),
        np.eye(4, 2)[[0, 1, 0, 1]],
        [{"weather": value} for value in ("sun", "rain", "sun", "rain")],
    )
    return Metadata(dataset)


@pytest.mark.required
class TestLegacyTargetLevel:
    """``level="target"`` still resolves on object detection, with a warning."""

    def test_rows_at_target(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'target' is deprecated"):
            rows = md.rows_at("target")  # type: ignore
        assert rows.height == 5
        assert rows["level"].unique().to_list() == ["instance"]

    def test_add_factors_at_target(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'target' is deprecated"):
            md.add_factors({"iou": [0.1, 0.2, 0.3, 0.4, 0.5]}, level="target")
        assert md.target_data["iou"].to_list() == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert md.factor_info["iou"].level == "instance"

    def test_ic_does_not_alias_target(self):
        """Image classification never reported "target", so it never accepts it."""
        md = _ic_metadata()
        md._structure()
        with pytest.raises(ValueError, match="Unknown level 'target'"):
            md.rows_at("target")  # type: ignore


@pytest.mark.required
class TestFactorInfoLevelRename:
    """The loud break: FactorInfo.level reports the real level name."""

    def test_reports_real_level_name(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            info = md.factor_info
        assert info["weather"].level == "image"

    def test_warns_once_per_instance(self, recwarn):
        md = _od_metadata()
        _ = md.factor_info
        _ = md.factor_info
        _ = md.filter_by_factor(lambda *_: False)
        renames = [w for w in recwarn.list if "FactorInfo.level now reports" in str(w.message)]
        assert len(renames) == 1

    def test_warns_from_filter_by_factor_too(self):
        """Every FactorInfo handout warns, not just the property."""
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="FactorInfo.level now reports"):
            md.filter_by_factor(lambda _, fi: fi.level == "instance")

    def test_ic_does_not_warn(self, recwarn):
        md = _ic_metadata()
        _ = md.factor_info
        _ = md.filter_by_factor(lambda *_: False)
        assert not [w for w in recwarn.list if "FactorInfo.level" in str(w.message)]

    def test_from_factors_does_not_warn(self, recwarn):
        """A factors-only instance has one level and never reported "target"."""
        md = Metadata.from_factors({"a": np.array([1, 2, 3])}, np.array([0, 1, 0]))
        _ = md.factor_info
        assert not [w for w in recwarn.list if "FactorInfo.level" in str(w.message)]


@pytest.mark.required
class TestRemovedInV1_2:
    """Members superseded by a general form, kept warning for one cycle."""

    def test_filter_by_factor_type(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="filter_by_factor_type.*removed in v1.2.0"):
            typed = md.filter_by_factor_type("categorical")
        np.testing.assert_array_equal(typed, md.filter_by_factor(lambda _, fi: fi.factor_type == "categorical"))

    def test_raw_data(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="raw_data.*removed in v1.2.0"):
            raw = md.raw_data
        np.testing.assert_array_equal(raw, md.target_data.select(md.factor_names).to_numpy())

    def test_get_image_factors(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="get_image_factors.*removed in v1.2.0"):
            assert md.get_image_factors(0)["weather"] == "sun"

    def test_get_target_factors(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="get_target_factors.*removed in v1.2.0"):
            assert md.get_target_factors(0, 1)["target_index"] == 1

    def test_image_data(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="image_data.*removed in v1.2.0"):
            assert md.image_data.equals(md.rows_at("image"))

    def test_image_data_returns_the_labelled_rows_for_classification(self):
        """Bug-for-bug with v1.1, where a classification dataset had one block of rows.

        Returning the image rows here would hand an existing caller nulls where
        ``class_label``/``score``/``target_index`` used to be.
        """
        md = _ic_metadata()
        with pytest.warns(DeprecationWarning, match="image_data.*removed in v1.2.0"):
            image_data = md.image_data

        assert image_data.equals(md.rows_at(md.label_level))
        assert image_data["class_label"].null_count() == 0

    def test_target_data(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="target_data.*removed in v1.2.0"):
            assert md.target_data.equals(md.rows_at(md.label_level))

    def test_has_targets(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="has_targets.*removed in v1.2.0"):
            assert md.has_targets() is True

    def test_has_targets_is_false_for_classification(self):
        """The v1.1 answer. Guards against reviving ``label_level != item_level``,
        which is true for every task now and would invert this."""
        md = _ic_metadata()
        with pytest.warns(DeprecationWarning, match="has_targets.*removed in v1.2.0"):
            assert md.has_targets() is False

    def test_has_targets_replacement_agrees_on_both_tasks(self):
        """The suggested replacement has to survive the cases that break count-based ones."""
        for build, expected in ((_od_metadata, True), (_ic_metadata, False)):
            md = build()
            with pytest.warns(DeprecationWarning, match="has_targets.*removed in v1.2.0"):
                assert md.has_targets() is md.multi_target is expected

    def test_target_factors_only_keeps_its_v1_1_semantics(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="target_factors_only.*removed in v1.2.0"):
            md.target_factors_only = True

        with pytest.warns(DeprecationWarning, match="target_factors_only.*removed in v1.2.0"):
            assert md.target_factors_only is True
        # Its own flag, not an alias for `inherited`, which has no task exemption.
        assert md.inherited is True

    def test_target_factors_only_is_a_no_op_for_classification(self):
        """v1.1 guarded this flag on ``has_targets``, so on IC it has never done anything.

        Forwarding it to ``inherited`` would instead empty the factor set, since IC puts
        essentially all per-item metadata at the image level.
        """
        md = _ic_metadata()
        before = list(md.factor_names)
        assert before

        with pytest.warns(DeprecationWarning, match="target_factors_only.*removed in v1.2.0"):
            md.target_factors_only = True

        assert list(md.factor_names) == before


@pytest.mark.required
class TestDeepCopy:
    """A Metadata carries a schema and a layout, and must stay copyable."""

    @pytest.mark.parametrize("build", [_ic_metadata, _od_metadata], ids=["IC", "OD"])
    def test_deepcopy_structures_identically(self, build):
        md = build()
        md._structure()

        clone = copy.deepcopy(md)

        assert clone.levels == md.levels
        assert clone.level_counts == md.level_counts
        assert clone.dataframe.equals(md.dataframe)
        np.testing.assert_array_equal(clone.class_labels, md.class_labels)
        np.testing.assert_array_equal(clone.item_indices, md.item_indices)

    @pytest.mark.parametrize("build", [_ic_metadata, _od_metadata], ids=["IC", "OD"])
    def test_copy_is_independent(self, build):
        """The regression this guards: deep-copy, then mutate, without touching the original."""
        md = build()
        clone = copy.deepcopy(md)
        clone.add_factors({"brightness": np.arange(clone.level_counts["image"], dtype=float)}, level="image")

        assert "brightness" in clone.dataframe.columns
        assert "brightness" not in md.dataframe.columns

    def test_deepcopy_before_structuring(self):
        md = _od_metadata()
        clone = copy.deepcopy(md)
        assert clone.levels == ("image", "instance")
        assert clone.dataframe.height == md.dataframe.height
