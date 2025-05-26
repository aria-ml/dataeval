import numpy as np
import polars as pl
import pytest

from dataeval.data._metadata import FactorInfo, Metadata
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

    def test_add_empty_factors(self):
        md = Metadata(None)  # type: ignore
        md._dataframe = pl.DataFrame()
        md._factors = {}
        md._count = 0
        md._is_structured = True
        md.add_factors({})
        assert md.factor_names == []

    def test_all_factor_types(self, RNG: np.random.Generator):
        md = Metadata(None)  # type: ignore
        md_dict = {
            "cat_str": RNG.choice(["A", "B"], size=100).tolist(),
            "cat_flt": RNG.choice([0.1, 0.2, 0.4, 0.6, 0.8], size=100).tolist(),
            "con_flt": RNG.random(size=100).tolist(),
            "dis_int": np.arange(100).tolist(),
        }
        md._dataframe = pl.from_dict(md_dict)
        md._factors = dict.fromkeys(md_dict, FactorInfo())
        md._is_structured = True

        md._bin()
        assert [f.factor_type for f in md.factor_info.values()] == [
            "categorical",
            "categorical",
            "continuous",
            "discrete",
        ]

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

    def test_raw_getter(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        raw_metadata = [{"foo": "bar"}]
        md._raw = raw_metadata
        assert md.raw == raw_metadata

    def test_empty_discretized_data(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._factors = {"foo": FactorInfo()}
        md._exclude = {"foo"}
        assert md.discretized_data.size == 0

    def test_empty_factor_data(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._factors = {"foo": FactorInfo()}
        md._exclude = {"foo"}
        assert md.factor_data.size == 0

    @pytest.mark.parametrize(
        "is_binned, exists",
        [
            (True, True),
            (True, False),
            (False, False),
        ],
    )
    def test_reset_bins(self, is_binned, exists):
        md = Metadata(None)  # type: ignore
        md._dataframe = pl.from_dict({"foo": [0], "foo[]": [0]} if exists else {"foo": [0]})
        md._factors = {"foo": FactorInfo("continuous", "foo[]" if exists else None)}
        md._is_binned = is_binned
        md._reset_bins()
        assert not md._is_binned
        assert "foo[]" not in md._dataframe.columns
        assert md._factors["foo"].discretized_col is None
