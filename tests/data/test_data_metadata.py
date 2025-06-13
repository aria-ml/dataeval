import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval.data._metadata import FactorInfo, Metadata, _binned
from dataeval.metrics.stats._boxratiostats import boxratiostats
from dataeval.metrics.stats._imagestats import imagestats
from dataeval.metrics.stats._labelstats import labelstats
from tests.data.test_data_embeddings import MockDataset


@dataclass
class ObjectDetectionTarget:
    boxes: Any
    labels: Any
    scores: Any


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

    def test_ic_empty_targets(self, mock_ds):
        mock_ds = copy.deepcopy(mock_ds)
        mock_ds.targets = list(mock_ds.targets)
        mock_ds.targets[1] = np.array([])
        mock_ds.targets[5] = np.array([])

        md = Metadata(mock_ds)
        assert len(md.class_labels) == 8
        assert md.image_indices.tolist() == [0, 2, 3, 4, 6, 7, 8, 9]

        stats = labelstats(md)
        assert stats.label_counts_per_image == [1, 0, 1, 1, 1, 0, 1, 1, 1, 1]

    def test_od_empty_targets(self, get_od_dataset):
        mock_ds = get_od_dataset(10, 2)
        for prop in ("_labels", "_bboxes"):
            _x = list(getattr(mock_ds, prop))
            _x[1] = []
            _x[5] = []
            setattr(mock_ds, prop, _x)

        md = Metadata(mock_ds)
        assert len(md.class_labels) == 16
        assert md.image_indices.tolist() == [0, 0, 2, 2, 3, 3, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9]

        stats = labelstats(md)
        assert stats.label_counts_per_image == [2, 0, 2, 2, 2, 0, 2, 2, 2, 2]

        imgstats = imagestats(mock_ds)
        boxstats = imagestats(mock_ds, per_box=True)
        ratiostats = boxratiostats(boxstats, imgstats)
        assert len(imgstats) == 10
        assert len(boxstats) == 16
        assert len(ratiostats) == 16

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
            "con_flt": RNG.random(size=100).tolist(),
            "dis_flt": RNG.choice([0.1, 0.2, 0.4, 0.6, 0.8], size=100).tolist(),
            "dis_int": np.arange(100).tolist(),
        }
        md._dataframe = pl.from_dict(md_dict)
        md._factors = dict.fromkeys(md_dict, None)
        md._is_structured = True
        md._image_indices = np.arange(100)

        md._bin()
        assert [f.factor_type for f in md.factor_info.values()] == [
            "categorical",
            "continuous",
            "discrete",
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

    def test_empty_digitized_data(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._factors = {"foo": None}
        md._exclude = {"foo"}
        assert md.digitized_data.size == 0

    def test_empty_binned_data(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._factors = {"foo": None}
        md._exclude = {"foo"}
        assert md.binned_data.size == 0

    def test_empty_factor_data(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._factors = {"foo": None}
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
        col = "foo"
        col_bn = _binned(col)
        md._dataframe = pl.from_dict({col: [0], col_bn: [0]} if exists else {col: [0]})
        md._factors = {col: FactorInfo("continuous", is_binned=exists)}
        md._is_binned = is_binned
        md._reset_bins()
        assert not md._is_binned
        assert col_bn not in md._dataframe.columns
        factor_info = md._factors[col]
        assert exists if factor_info is None else not factor_info.is_binned
