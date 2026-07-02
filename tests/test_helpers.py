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
