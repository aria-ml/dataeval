from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.data._metadata import Metadata
from dataeval.utils.datasets._types import ObjectDetectionTarget
from tests.data.test_data_embeddings import MockDataset


@pytest.fixture(scope="module")
def mock_ds() -> MockDataset:
    return MockDataset(np.ones((10, 3, 3)), np.ones((10, 3)), [{str(i): i} for i in range(10)])


@pytest.fixture(scope="module")
def mock_metadata(mock_ds) -> Metadata:
    return Metadata(mock_ds)


@pytest.mark.required
class TestMetadata:
    """
    Test collate aggregates MAITE style data into separate collections from tuple return
    """

    @pytest.mark.parametrize(
        "data, labels, metadata, factors",
        [
            (np.ones((10, 3, 3)), np.ones((10, 3)), [{str(i): i} for i in range(10)], 10),
            (
                np.ones((10, 3, 3)),
                [ObjectDetectionTarget([[0, 1, 2, 3], [4, 5, 6, 7]], [0, 1], [1, 0]) for _ in range(10)],
                [{str(i): i} for i in range(10)],
                10,
            ),
            (
                np.ones((10, 3, 3)),
                [ObjectDetectionTarget([[0, 1, 2, 3], [4, 5, 6, 7]], [0, 1], [1, 0]) for _ in range(10)],
                [{str(i): i} for i in range(10)],
                20,
            ),
        ],
    )
    def test_mock_inputs(self, data, labels, metadata, factors):
        """Tests common (input, target, metadata) dataset output"""
        ds = MockDataset(data, labels, metadata)
        md = Metadata(ds)

        md.add_factors({"a": np.random.random((factors,))})
        assert "a" in md.discrete_factor_names
        assert "a" in md.merged

    def test_mismatch_factor_length(self, mock_metadata):
        with pytest.raises(ValueError, match="provided factors have a different length"):
            mock_metadata.add_factors({"a": np.random.random((20,))})

    def test_exclude_no_op(self):
        md = Metadata(None, exclude=["a", "b"])  # type: ignore
        md._processed = True
        md.exclude = ["b", "a"]
        assert md._processed

    def test_include_no_op(self):
        md = Metadata(None, include=["a", "b"])  # type: ignore
        md._processed = True
        md.include = ["b", "a"]
        assert md._processed

    def test_exclude_and_include_both_provided(self):
        with pytest.raises(ValueError, match="Filters for `exclude` and `include` are mutually exclusive."):
            Metadata(None, exclude=["a"], include=["b"])  # type: ignore

    def test_exclude_setter_from_include(self):
        md = Metadata(None, include=["a"])  # type: ignore
        md._processed = True
        assert not md._exclude
        md.exclude = ["b"]
        assert md._exclude == {"b"}
        assert not md._include
        assert not md._processed

    def test_exclude_setter_from_exclude(self):
        md = Metadata(None, exclude=["a"])  # type: ignore
        md._processed = True
        assert md._exclude == {"a"}
        md.exclude = ["b"]
        assert md._exclude == {"b"}
        assert not md._processed

    def test_include_setter_from_include(self):
        md = Metadata(None, include=["a"])  # type: ignore
        md._processed = True
        assert md._include == {"a"}
        md.include = ["b"]
        assert md._include == {"b"}
        assert not md._processed

    def test_include_setter_from_exclude(self):
        md = Metadata(None, exclude=["a"])  # type: ignore
        md._processed = True
        assert not md._include
        md.include = ["b"]
        assert md._include == {"b"}
        assert not md._exclude
        assert not md._processed

    def test_dropped_factors(self):
        md = Metadata(None)  # type: ignore
        md._processed = True
        md._merged = ({"a": [1]}, {"b": ["foo"], "c": ["bar"]})
        assert md.dropped_factors == {"b": ["foo"], "c": ["bar"]}

    def test_unknown_target(self):
        md = Metadata([(np.zeros((3, 16, 16)), "THIS IS NOT A TARGET", {"id": 0})])  # type: ignore
        with pytest.raises(TypeError, match="Encountered unsupported target type in dataset"):
            md._collate()

    def test_mixed_target(self):
        md = Metadata(
            [
                (np.zeros((3, 16, 16)), np.zeros((3,)), {"id": 0}),
                (np.zeros((3, 16, 16)), ObjectDetectionTarget([[0, 0, 0, 0]], [0], [0, 0, 0]), {"id": 0}),
            ]  # type: ignore
        )
        with pytest.raises(ValueError, match="Encountered unexpected target type in dataset"):
            md._collate()

    def test_validate_unknown_merged_values(self):
        md = Metadata(None)  # type: ignore
        md._targets = MagicMock()
        md._targets.labels.ndim = 1
        md._merged = ({"a": "not a list tuple or array"}, {"b": ["foo"], "c": ["bar"]})  # type: ignore
        with pytest.raises(TypeError, match="values are arraylike"):
            md._validate()

    def test_process_include(self, mock_ds):
        md = Metadata(mock_ds, include=["id"])
        md._process()

    def test_process_exclude(self, mock_ds):
        md = Metadata(mock_ds, exclude=["id"])
        md._process()

    def test_contiguous_factor_bins_setter(self):
        md = Metadata(None)  # type: ignore
        md._processed = True
        md.continuous_factor_bins = {"a": 10}
        assert not md._processed
        assert md._continuous_factor_bins == {"a": 10}

    def test_contiguous_factor_bins_setter_no_op(self):
        md = Metadata(None, continuous_factor_bins={"a": 10})  # type: ignore
        md._processed = True
        md.continuous_factor_bins = {"a": 10}
        assert md._processed
        assert md._continuous_factor_bins == {"a": 10}

    def test_auto_bin_method_setter(self):
        md = Metadata(None)  # type: ignore
        md._processed = True
        md.auto_bin_method = "clusters"
        assert not md._processed
        assert md._auto_bin_method == "clusters"

    def test_auto_bin_method_setter_no_op(self):
        md = Metadata(None, auto_bin_method="clusters")  # type: ignore
        md._processed = True
        md.auto_bin_method = "clusters"
        assert md._processed
        assert md._auto_bin_method == "clusters"
