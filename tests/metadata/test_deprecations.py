"""The compatibility surface left behind by the level restructure."""

import copy
import inspect
import warnings

import numpy as np
import pytest

from dataeval._metadata import Metadata
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget
from tests.metadata.test_structurers import _SHAPES, _mot_dataset


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
        assert info["weather"].level == "unit"

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
            assert md.image_data.equals(md.rows_at("unit"))

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
        clone.add_factors({"brightness": np.arange(clone.level_counts["unit"], dtype=float)}, level="unit")

        assert "brightness" in clone.dataframe.columns
        assert "brightness" not in md.dataframe.columns

    def test_deepcopy_before_structuring(self):
        md = _od_metadata()
        clone = copy.deepcopy(md)
        assert clone.levels == ("unit", "instance")
        assert clone.dataframe.height == md.dataframe.height


@pytest.mark.required
class TestLegacyImageLevel:
    """``"image"`` still resolves to the ``unit`` level, with a warning, on every task."""

    def test_rows_at_image(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            rows = md.rows_at("image")  # type: ignore[arg-type]
        assert rows["level"].unique().to_list() == ["unit"]

    def test_at_image(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            view = md.at("image")  # type: ignore[arg-type]
        assert view.view == "unit"

    def test_view_setter_image(self):
        md = _ic_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            md.view = "image"  # type: ignore[assignment]
        assert md.view == "unit"

    def test_add_factors_at_image(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            md.add_factors({"blur": [0.1, 0.2, 0.3]}, level="image")  # type: ignore[arg-type]
        assert md.factor_info["blur"].level == "unit"

    def test_constructor_view_image(self):
        """A view chosen at construction resolves when the schema first exists."""
        dataset = MockDataset(
            np.zeros((4, 3, 16, 16)),
            np.eye(4, 2)[[0, 1, 0, 1]],
            [{"weather": v} for v in ("sun", "rain", "sun", "rain")],
        )
        md = Metadata(dataset, view="image")  # type: ignore[arg-type]
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            assert md.view == "unit"

    def test_from_factors_level_image(self):
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            md = Metadata.from_factors({"a": np.array([0, 1, 0])}, level="image")  # type: ignore[arg-type]
        assert md.levels == ("unit",)

    def test_mot_also_aliases_image(self):
        """MOT's units are frames, and a frame is an image; the alias is universal."""
        md = Metadata(_mot_dataset(_SHAPES))
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            rows = md.rows_at("image")  # type: ignore[arg-type]
        assert rows["level"].unique().to_list() == ["unit"]

    def test_od_still_aliases_target_too(self):
        """Adding the image alias must not displace the existing target alias."""
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="Level 'target' is deprecated"):
            assert md.rows_at("target").height == 5  # type: ignore[arg-type]


@pytest.mark.required
class TestFactorInfoLevelRenameWarning:
    """Every task now announces that ``FactorInfo.level`` stopped saying "image"."""

    def test_ic_now_warns(self):
        md = _ic_metadata()
        with pytest.warns(DeprecationWarning, match="no longer reports"):
            _ = md.factor_info

    def test_mot_now_warns(self):
        md = Metadata(_mot_dataset(_SHAPES, [{"weather": "sun"}, {"weather": "rain"}]))
        with pytest.warns(DeprecationWarning, match="no longer reports"):
            _ = md.factor_info

    def test_still_once_per_instance(self, recwarn):
        md = _ic_metadata()
        _ = md.factor_info
        recwarn.clear()
        _ = md.factor_info
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


@pytest.mark.required
class TestFactorsOnlyAliasesImageOnlyAtUnitLevel:
    """A factors-only instance below the unit level has no ``"image"`` to retire.

    The alias map is declared on ``Structurer`` because every *task* has a unit level for
    ``"image"`` to resolve to. ``from_factors(level="instance")`` does not, and inheriting
    the alias unconditionally produced advice that could never apply: a rename warning
    about rows that are not units, and a warn-then-raise out of ``rows_at("image")``.
    """

    def test_instance_level_factor_info_does_not_warn(self, recwarn):
        md = Metadata.from_factors({"weather": np.array([0, 1, 0, 1])}, level="instance")
        assert md.factor_info["weather"].level == "instance"
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_instance_level_rejects_image_without_warning_first(self, recwarn):
        """The retired spelling is simply not a level here, and is refused as one."""
        md = Metadata.from_factors({"weather": np.array([0, 1, 0, 1])}, level="instance")
        with pytest.raises(ValueError, match="Unknown level 'image'"):
            md.rows_at("image")  # type: ignore[arg-type]
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    @pytest.mark.parametrize(
        "build",
        [
            lambda factors: Metadata.from_factors(factors),
            lambda factors: Metadata.from_factors(factors, level="unit"),
        ],
        ids=["default", "explicit-unit"],
    )
    def test_unit_level_still_warns_and_still_resolves(self, build):
        md = build({"weather": np.array([0, 1, 0, 1])})
        with pytest.warns(DeprecationWarning, match="no longer reports"):
            assert md.factor_info["weather"].level == "unit"
        with pytest.warns(DeprecationWarning, match="Level 'image' is deprecated"):
            assert md.rows_at("image")["level"].unique().to_list() == ["unit"]  # type: ignore[arg-type]


def _first_deprecation(caught: list[warnings.WarningMessage]) -> warnings.WarningMessage:
    matches = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(matches) == 1, f"expected exactly one DeprecationWarning, got {matches}"
    return matches[0]


@pytest.mark.required
class TestDeprecationStacklevel:
    """``pytest.warns`` checks a warning's category and message but never its location,
    so a ``stacklevel`` regression is invisible to every other test in this module. Each
    of these instead records the warning and asserts the attributed frame directly.

    Regression coverage for a bug where extracting the warning into a shared helper
    (``_resolve_legacy_level``) added a stack frame that none of ``_resolve_level``'s
    callers compensated for, so every one of these blamed a line inside ``dataeval``
    instead of the caller.
    """

    def test_view_setter_blames_caller(self):
        md = _ic_metadata()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md.view = "image"  # type: ignore[assignment]
        assert _first_deprecation(caught).filename == __file__

    def test_rows_at_blames_caller(self):
        md = _od_metadata()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md.rows_at("image")  # type: ignore[arg-type]
        assert _first_deprecation(caught).filename == __file__

    def test_at_blames_caller(self):
        md = _od_metadata()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md.at("image")  # type: ignore[arg-type]
        assert _first_deprecation(caught).filename == __file__

    def test_add_factors_blames_caller(self):
        md = _od_metadata()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md.add_factors({"blur": [0.1, 0.2, 0.3]}, level="image")  # type: ignore[arg-type]
        assert _first_deprecation(caught).filename == __file__

    def test_from_factors_blames_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Metadata.from_factors({"a": np.array([0, 1, 0])}, level="image")  # type: ignore[arg-type]
        assert _first_deprecation(caught).filename == __file__

    def test_target_alias_blames_caller_too(self):
        """The pre-existing "target" alias runs through the same helper and shared the bug."""
        md = _od_metadata()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md.rows_at("target")  # type: ignore[arg-type]
        assert _first_deprecation(caught).filename == __file__

    def test_constructor_view_blames_adopt_not_resolve_level(self):
        """A view chosen at construction can't blame the constructor call — that frame
        is long gone by the time structuring lazily happens — so this cannot assert the
        caller's line the way the others do. It instead guards against regressing to the
        pre-fix location, which was inside ``_resolve_level``'s own body: the least
        useful line in the whole call chain, since every caller of it lands there.
        """
        dataset = MockDataset(
            np.zeros((4, 3, 16, 16)),
            np.eye(4, 2)[[0, 1, 0, 1]],
            [{"weather": v} for v in ("sun", "rain", "sun", "rain")],
        )
        md = Metadata(dataset, view="image")  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = md.view
        w = _first_deprecation(caught)

        lines, start = inspect.getsourcelines(Metadata._adopt)
        end = start + len(lines) - 1
        assert w.filename.endswith("_metadata.py")
        assert start <= w.lineno <= end, (
            f"expected the warning inside Metadata._adopt (lines {start}-{end}), got line {w.lineno}"
        )


@pytest.mark.required
class TestLevelMessagesNameTheUnitType:
    """An unknown or retired level is explained in the dataset's own vocabulary."""

    def test_unknown_level_names_the_unit_type(self):
        md = _od_metadata()
        with pytest.raises(ValueError, match="this dataset's units are images"):
            md.rows_at("frame")  # type: ignore[arg-type]

    def test_unknown_level_on_mot_names_frames(self):
        md = Metadata(_mot_dataset(_SHAPES))
        with pytest.raises(ValueError, match="this dataset's units are frames"):
            md.rows_at("clip")  # type: ignore[arg-type]

    def test_unknown_level_still_lists_the_levels(self):
        md = _od_metadata()
        with pytest.raises(ValueError, match="Available levels are"):
            md.rows_at("frame")  # type: ignore[arg-type]

    def test_image_deprecation_names_the_unit_type(self):
        md = _od_metadata()
        with pytest.warns(DeprecationWarning, match="this dataset's units are images"):
            md.rows_at("image")  # type: ignore[arg-type]

    def test_from_factors_deprecation_names_items(self):
        """``_load_factors`` has no structurer instance yet to ask, so it resolves
        against ``Structurer``'s own base-class ``unit_type`` ("item") rather than a
        task-specific one — this is the only path that exercises that default.
        """
        factors = {"weather": np.array([0, 1, 0, 1])}
        with pytest.warns(DeprecationWarning, match="this dataset's units are items"):
            Metadata.from_factors(factors, level="image")  # type: ignore[arg-type]

    def test_unknown_level_on_factors_only_names_items(self):
        factors = {"weather": np.array([0, 1, 0, 1])}
        md = Metadata.from_factors(factors)
        with pytest.raises(ValueError, match="this dataset's units are items"):
            md.rows_at("bogus")  # type: ignore[arg-type]
