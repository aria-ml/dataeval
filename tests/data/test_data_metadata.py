import numpy as np
import polars as pl
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
        assert "a" in md.factor_names
        assert "a" in md.dataframe

    def test_mismatch_factor_length(self, mock_metadata):
        with pytest.raises(ValueError, match="provided factors have a different length"):
            mock_metadata.add_factors({"a": np.random.random((20,))})

    def test_exclude_no_op(self):
        md = Metadata(None, exclude=["a", "b"])  # type: ignore
        md._is_binned = True
        md.exclude = ["b", "a"]
        assert md._is_binned

    def test_include_no_op(self):
        md = Metadata(None, include=["a", "b"])  # type: ignore
        md._is_binned = True
        md.include = ["b", "a"]
        assert md._is_binned

    def test_exclude_and_include_both_provided(self):
        with pytest.raises(ValueError, match="Filters for `exclude` and `include` are mutually exclusive."):
            Metadata(None, exclude=["a"], include=["b"])  # type: ignore

    def test_exclude_setter_from_include(self):
        md = Metadata(None, include=["a"])  # type: ignore
        md._is_binned = True
        assert not md._exclude
        md.exclude = ["b"]
        assert md._exclude == {"b"}
        assert not md._include
        assert not md._is_binned

    def test_exclude_setter_from_exclude(self):
        md = Metadata(None, exclude=["a"])  # type: ignore
        md._is_binned = True
        assert md._exclude == {"a"}
        md.exclude = ["b"]
        assert md._exclude == {"b"}
        assert not md._is_binned

    def test_include_setter_from_include(self):
        md = Metadata(None, include=["a"])  # type: ignore
        md._is_binned = True
        assert md._include == {"a"}
        md.include = ["b"]
        assert md._include == {"b"}
        assert not md._is_binned

    def test_include_setter_from_exclude(self):
        md = Metadata(None, exclude=["a"])  # type: ignore
        md._is_binned = True
        assert not md._include
        md.include = ["b"]
        assert md._include == {"b"}
        assert not md._exclude
        assert not md._is_binned

    def test_dropped_factors(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._dropped_factors = {"b": ["foo"], "c": ["bar"]}
        assert md.dropped_factors == {"b": ["foo"], "c": ["bar"]}

    def test_unknown_target(self):
        md = Metadata([(np.zeros((3, 16, 16)), "THIS IS NOT A TARGET", {"id": 0})])  # type: ignore
        with pytest.raises(TypeError, match="Encountered unsupported target type in dataset"):
            md._structure()

    def test_mixed_target(self):
        md = Metadata(
            [
                (np.zeros((3, 16, 16)), np.zeros((3,)), {"id": 0}),
                (np.zeros((3, 16, 16)), ObjectDetectionTarget([[0, 0, 0, 0]], [0], [0, 0, 0]), {"id": 0}),
            ]  # type: ignore
        )
        with pytest.raises(ValueError, match="Encountered unexpected target type in dataset"):
            md._structure()

    def test_process_include(self, mock_ds):
        md = Metadata(mock_ds, include=["id"])
        md._bin()

    def test_process_exclude(self, mock_ds):
        md = Metadata(mock_ds, exclude=["id"])
        md._bin()

    def test_contiguous_factor_bins_setter(self):
        md = Metadata(None)  # type: ignore
        md._dataframe = pl.DataFrame()
        md._is_binned = True
        md.continuous_factor_bins = {"a": 10}
        assert not md._is_binned
        assert md._continuous_factor_bins == {"a": 10}

    def test_contiguous_factor_bins_setter_no_op(self):
        md = Metadata(None, continuous_factor_bins={"a": 10})  # type: ignore
        md._dataframe = pl.DataFrame()
        md._is_binned = True
        md.continuous_factor_bins = {"a": 10}
        assert md._is_binned
        assert md._continuous_factor_bins == {"a": 10}

    def test_auto_bin_method_setter(self):
        md = Metadata(None)  # type: ignore
        md._dataframe = pl.DataFrame()
        md._is_binned = True
        md.auto_bin_method = "clusters"
        assert not md._is_binned
        assert md._auto_bin_method == "clusters"

    def test_auto_bin_method_setter_no_op(self):
        md = Metadata(None, auto_bin_method="clusters")  # type: ignore
        md._dataframe = pl.DataFrame()
        md._is_binned = True
        md.auto_bin_method = "clusters"
        assert md._is_binned
        assert md._auto_bin_method == "clusters"
