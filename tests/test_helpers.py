"""Tests for dataeval._helpers config utilities."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel

from dataeval import Metadata
from dataeval._helpers import apply_config, axis_vocabulary, get_overrides, independence_groups
from tests.bias.test_balance import _sequences
from tests.conftest import MockMetadata, to_metadata


class _Config(BaseModel):
    a: int = 1
    b: int = 2


class _Target:
    pass


@pytest.mark.required
class TestApplyConfig:
    def test_applies_all_fields_by_default(self):
        obj = _Target()
        apply_config(obj, _Config())
        assert getattr(obj, "a", None) == 1
        assert getattr(obj, "b", None) == 2
        assert getattr(obj, "config", None) is not None

    def test_excluded_field_is_skipped(self):
        # 'b' is in the exclude set, so it takes the loop's skip branch and is not set
        obj = _Target()
        apply_config(obj, _Config(), exclude={"b"})
        assert getattr(obj, "a", None) == 1
        assert not hasattr(obj, "b")


@pytest.mark.required
class TestGetOverrides:
    def test_drops_ignored_and_none_values(self):
        overrides = get_overrides({"self": object(), "config": None, "a": 1, "b": None})
        assert overrides == {"a": 1}


@pytest.mark.required
class TestIsMetadataLike:
    """Dispatch must recognize Metadata without touching its properties.

    ``MetadataLike`` is a runtime_checkable protocol. Python 3.12+ resolves protocol
    members with ``inspect.getattr_static``, but 3.11 uses ``hasattr``, which
    calls property getters — so a bare ``isinstance`` against the protocol structures
    and bins the whole dataset inside a type check, and raises rather than returning
    False at a view where ``class_labels`` is undefined. Both versions are supported,
    so these assert the behavior that has to hold on all of them.
    """

    def test_metadata_is_recognized_without_structuring(self, get_od_dataset):
        from dataeval import Metadata
        from dataeval._helpers import is_metadata_like

        metadata = Metadata(get_od_dataset(6, targets_per_image=2))
        assert is_metadata_like(metadata)
        assert not metadata._is_structured
        assert not metadata._is_binned

    def test_view_above_label_level_does_not_raise(self, get_od_dataset):
        # class_labels raises above label_level by design; on 3.11 hasattr only
        # swallows AttributeError, so that ValueError used to escape isinstance itself.
        from dataeval import Metadata
        from dataeval._helpers import is_metadata_like

        assert is_metadata_like(Metadata(get_od_dataset(6, targets_per_image=2)).at("unit"))

    def test_third_party_container_still_resolves(self, recwarn):
        from dataeval._helpers import is_metadata_like

        class Simple:
            factor_names = ["a"]
            factor_data = [[0]]
            class_labels = [0]
            is_binned = [False]

        assert is_metadata_like(Simple())
        assert not [w for w in recwarn.list if "is_binned" in str(w.message)]

    def test_container_with_neither_member_is_rejected(self):
        from dataeval._helpers import is_metadata_like

        class Neither:
            factor_names = ["a"]
            factor_data = [[0]]
            class_labels = [0]

        assert not is_metadata_like(Neither())

    def test_non_metadata_objects_are_rejected(self, get_od_dataset):
        from dataeval._helpers import is_metadata_like

        assert not is_metadata_like(object())
        assert not is_metadata_like(get_od_dataset(2))


@pytest.mark.required
class TestAxisVocabulary:
    def test_an_axis_with_no_components_is_observed(self):
        """Nothing a dataset could have been declared against, so nothing is declared."""
        assert axis_vocabulary([]) == "observed"

    def test_a_single_component_with_no_spec_is_observed(self):
        assert axis_vocabulary([None]) == "observed"


@pytest.mark.required
class TestEffectiveEntityCounts:
    def test_aborts_when_a_factor_level_cannot_be_determined(self):
        """A factor whose hierarchy level is unknown aborts the effective-n calculation."""
        from types import SimpleNamespace

        from dataeval._helpers import effective_entity_counts

        metadata = SimpleNamespace(
            _entity_counts=lambda levels: [1, 1],
            factor_info={"w": SimpleNamespace(level=None)},
        )
        assert effective_entity_counts(metadata, SimpleNamespace(level="unit"), ["w"]) is None  # type: ignore[arg-type]


def _bare(item_indices=None, n=5):
    """A container with no levels of its own, optionally declaring ``item_indices``."""
    return MockMetadata(
        class_labels=np.zeros(n, dtype=np.intp),
        factor_data=np.zeros((n, 1), dtype=np.int64),
        factor_names=["a"],
        is_binned=[False],
        index2label={0: "x"},
        item_indices=None if item_indices is None else np.asarray(item_indices),
    )


@pytest.mark.required
class TestIndependenceGroups:
    """Parity's independent sampling unit: recorded structure first, the caller's word last."""

    def test_od_groups_by_image(self, get_od_dataset):
        md = Metadata(get_od_dataset(6, targets_per_image=3))
        result = independence_groups(md, assume_independent=False)
        assert result.source == "level"
        assert result.level == "unit"
        assert result.groups is not None
        np.testing.assert_array_equal(result.groups, md.item_indices)
        assert len(np.unique(result.groups)) == 6

    def test_tracking_groups_by_sequence(self):
        md = Metadata(_sequences(4, 5))
        md._structure()
        result = independence_groups(md, assume_independent=False)
        assert result.source == "level"
        assert result.level == "sequence"
        assert result.groups is not None
        assert len(result.groups) == len(md.item_indices)
        assert sorted(np.bincount(result.groups)) == [5, 5, 5, 5]

    def test_frame_view_groups_by_sequence(self):
        """A view between the rows and the sequences is still clustered by sequence."""
        md = Metadata(_sequences(4, 5))
        md._structure()
        frames = md.at("unit")
        result = independence_groups(frames, assume_independent=False)
        assert result.source == "level"
        assert result.level == "sequence"
        assert result.groups is not None
        assert len(result.groups) == len(frames.item_indices) == md.level_counts["unit"]
        assert len(np.unique(result.groups)) == 4

    def test_ic_and_item_view_are_rows(self, get_od_dataset):
        ic = to_metadata({"a": [0, 1, 0, 1]}, [0, 1, 0, 1])
        od = Metadata(get_od_dataset(6, targets_per_image=3))
        od._structure()
        for md in (ic, od.at(od.item_level)):
            result = independence_groups(md, assume_independent=False)
            assert result.source == "rows"
            assert result.groups is None
            assert result.level is None

    def test_from_factors_with_repeated_item_ids_groups_by_item(self):
        """A Metadata at its item-level view still groups by item ids that repeat across rows."""
        factors, labels = {"a": [0, 1, 0, 1, 0]}, [0, 1, 0, 1, 1]
        md = Metadata.from_factors(factors, class_labels=labels, item_indices=np.array([0, 0, 1, 1, 2]))
        result = independence_groups(md, assume_independent=False)
        assert result.source == "level"
        assert result.level == md.item_level
        assert result.groups is not None
        np.testing.assert_array_equal(result.groups, [0, 0, 1, 1, 2])

        unique = Metadata.from_factors(factors, class_labels=labels, item_indices=np.arange(5))
        assert independence_groups(unique, assume_independent=False).source == "rows"

    def test_custom_item_indices(self):
        result = independence_groups(_bare([0, 0, 1, 1, 2]), assume_independent=False)
        assert result.source == "item"
        assert result.level is None
        assert result.groups is not None
        np.testing.assert_array_equal(result.groups, [0, 0, 1, 1, 2])

        one_to_one = independence_groups(_bare([4, 2, 0, 1, 3]), assume_independent=False)
        assert one_to_one.source == "rows"
        assert one_to_one.groups is None

    def test_bare_container_refuses_with_the_fix(self):
        with pytest.raises(ValueError, match="assume_independent=True"):
            independence_groups(_bare(), assume_independent=False)

    def test_declared_only_when_nothing_is_recorded(self, get_od_dataset):
        declared = independence_groups(_bare(), assume_independent=True)
        assert (declared.source, declared.groups, declared.level) == ("declared", None, None)

        # The recorded structure wins over the caller's word, for a Metadata and for item_indices.
        od = Metadata(get_od_dataset(6, targets_per_image=3))
        assert independence_groups(od, assume_independent=True).source == "level"
        assert independence_groups(_bare([0, 0, 1, 1, 2]), assume_independent=True).source == "item"
        assert independence_groups(_bare([0, 1, 2, 3, 4]), assume_independent=True).source == "rows"

    @pytest.mark.parametrize("indices", [[0, 1, 2], [], [0, 0, 1], [0, 1, 2, 3, 4, 5]])
    def test_item_indices_not_matching_the_rows_are_refused(self, indices):
        """A malformed record must refuse, not read as unique and so certify independence."""
        with pytest.raises(ValueError, match=f"{len(indices)} entries for 5 rows"):
            independence_groups(_bare(indices), assume_independent=True)

    def test_metadata_item_indices_not_matching_the_rows_are_refused(self):
        class Stub:
            item_level = "unit"
            view = "instance"
            item_indices = np.array([0, 1, 2])
            _levels = SimpleNamespace(parents_of=lambda level: ())

            def __len__(self):
                return 5

        with pytest.raises(ValueError, match="3 entries for 5 rows"):
            independence_groups(Stub(), assume_independent=True)

    def test_item_level_with_parents_raises(self, get_od_dataset):
        """No structurer builds a level above the items; one that did would be guessed at."""
        md = Metadata(get_od_dataset(6, targets_per_image=3))
        md._structure()
        stub = SimpleNamespace(
            item_level=md.item_level,
            view=md.view,
            item_indices=md.item_indices,
            _levels=SimpleNamespace(parents_of=lambda level: ("above",)),
        )
        with pytest.raises(ValueError, match="a level above the item level"):
            independence_groups(stub, assume_independent=True)
