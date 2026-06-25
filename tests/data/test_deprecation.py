"""Verify the backward-compat forwarding shims for the dataeval.data move."""

import warnings

import pytest

import dataeval.data

_SELECTION_NAMES = ("ClassBalance", "ClassFilter", "Indices", "Limit", "Reverse", "Select", "Selection", "Shuffle")
_UTILS_DATA_OPS = ("split_dataset", "unzip_dataset", "TrainValSplit", "DatasetSplits")


@pytest.mark.required
class TestSelectionShim:
    def test_from_import_warns_and_forwards(self):
        with pytest.warns(DeprecationWarning, match="dataeval.data"):
            from dataeval.selection import Select
        assert Select is dataeval.data.Select

    def test_all_moved_names_forward_to_same_object(self):
        import dataeval.selection as legacy

        for name in _SELECTION_NAMES:
            with pytest.warns(DeprecationWarning, match="has moved to dataeval.data"):
                obj = getattr(legacy, name)
            assert obj is getattr(dataeval.data, name)

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
