"""Tests for dataeval._helpers config utilities."""

import pytest
from pydantic import BaseModel

from dataeval._helpers import apply_config, get_overrides


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
    members with ``inspect.getattr_static``, but 3.10 and 3.11 use ``hasattr``, which
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
        # class_labels raises above label_level by design; on 3.10/3.11 hasattr only
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
