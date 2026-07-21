"""Verify the backward-compat forwarding shims for the dataeval.data move."""

import warnings

import pytest

import dataeval.data

_MOVED_OPS = ("ClassBalance", "ClassFilter", "Indices", "Limit", "Reverse", "Shuffle")
_UTILS_DATA_OPS = ("split_dataset", "unzip_dataset", "TrainValSplit", "DatasetSplits")


class _MiniDataset:
    metadata = {"id": "ds"}

    def __init__(self, n: int = 6) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> int:
        return i


@pytest.mark.required
class TestSelectViewDeprecation:
    """dataeval.selection.Select/Selector are deprecated shims for View/Operation."""

    def test_select_instantiation_warns_and_works(self):
        from dataeval.data import Limit, View
        from dataeval.selection import Select

        with pytest.warns(DeprecationWarning, match="View"):
            s = Select(_MiniDataset(4), selections=[Limit(2)])  # type: ignore
        assert isinstance(s, View)
        assert len(s) == 2
        assert s.selection_groups == s.operation_groups

    def test_selector_is_deprecated_alias_for_operation(self):
        import dataeval.data
        from dataeval.data import View

        with pytest.warns(DeprecationWarning, match="Operation"):
            from dataeval.selection import Selector
        assert Selector is dataeval.data.Operation

        class Evens(Selector):  # new contract: implement apply(view)
            def apply(self, view):
                view.selection = [i for i in view.selection if i % 2 == 0]

        v = View(_MiniDataset(6), [Evens()])  # type: ignore
        assert list(v) == [0, 2, 4]


@pytest.mark.required
class TestSelectionShim:
    def test_from_import_warns_and_forwards(self):
        with pytest.warns(DeprecationWarning, match="dataeval.data"):
            from dataeval.selection import ClassFilter
        assert ClassFilter is dataeval.data.ClassFilter

    def test_all_moved_ops_forward_to_same_object(self):
        import dataeval.selection as legacy

        for name in _MOVED_OPS:
            with pytest.warns(DeprecationWarning, match="has moved to dataeval.data"):
                obj = getattr(legacy, name)
            assert obj is getattr(dataeval.data, name)

    def test_selection_and_selector_map_to_operation(self):
        import dataeval.selection as legacy

        for name in ("Selection", "Selector"):
            with pytest.warns(DeprecationWarning, match="deprecated"):
                obj = getattr(legacy, name)
            assert obj is dataeval.data.Operation

    def test_unknown_attribute_raises(self):
        import dataeval.selection as legacy

        with pytest.raises(AttributeError):
            _ = legacy.DoesNotExist


@pytest.mark.required
class TestUtilsDataShim:
    def test_moved_ops_warn_and_forward_to_same_object(self):
        import dataeval.utils.data as legacy

        for name in _UTILS_DATA_OPS:
            with pytest.warns(DeprecationWarning, match="has moved to dataeval.data"):
                obj = getattr(legacy, name)
            assert obj is getattr(dataeval.data, name)

    def test_validation_helpers_remain_without_deprecation(self):
        # DatasetKind / validate_dataset / requires_maite_dataset are validation infra, not
        # data ops — they stay importable from dataeval.utils.data with no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            from dataeval.utils.data import DatasetKind, requires_maite_dataset, validate_dataset

        assert DatasetKind is not None
        assert validate_dataset is not None
        assert requires_maite_dataset is not None

    def test_unknown_attribute_raises(self):
        import dataeval.utils.data as legacy

        with pytest.raises(AttributeError):
            _ = legacy.DoesNotExist
