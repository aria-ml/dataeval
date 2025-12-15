import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval.core import calculate
from dataeval.core._calculate_ratios import calculate_ratios
from dataeval.core._label_stats import label_stats
from dataeval.core.flags import ImageStats
from dataeval.data._metadata import FactorInfo, Metadata, _binned
from dataeval.utils import unzip_dataset
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
        assert md.item_indices.tolist() == [0, 2, 3, 4, 6, 7, 8, 9]

        # Extract labels from metadata
        labels: list[list[int]] = [[] for _ in range(md.item_count)]
        for class_label, item_index in zip(md.class_labels, md.item_indices):
            labels[item_index].append(int(class_label))
        stats = label_stats(labels, md.index2label)

        assert stats["label_counts_per_image"] == [1, 0, 1, 1, 1, 0, 1, 1, 1, 1]

    def test_od_empty_targets(self, get_od_dataset):
        mock_ds = get_od_dataset(10, 2)
        for prop in ("_labels", "_bboxes"):
            _x = list(getattr(mock_ds, prop))
            _x[1] = []
            _x[5] = []
            setattr(mock_ds, prop, _x)

        md = Metadata(mock_ds)
        assert len(md.class_labels) == 16
        assert md.item_indices.tolist() == [0, 0, 2, 2, 3, 3, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9]

        # Extract labels from metadata
        labels: list[list[int]] = [[] for _ in range(md.item_count)]
        for class_label, item_index in zip(md.class_labels, md.item_indices):
            labels[item_index].append(int(class_label))
        stats = label_stats(labels, md.index2label)

        assert stats["label_counts_per_image"] == [2, 0, 2, 2, 2, 0, 2, 2, 2, 2]

        imgstats = calculate(
            *unzip_dataset(mock_ds, False), stats=ImageStats.PIXEL | ImageStats.VISUAL, per_image=True, per_target=False
        )
        boxstats = calculate(
            *unzip_dataset(mock_ds, True), stats=ImageStats.PIXEL | ImageStats.VISUAL, per_image=False, per_target=True
        )
        ratiostats = calculate_ratios(imgstats, target_stats_output=boxstats)
        assert len(imgstats["source_index"]) == 10
        assert len(boxstats["source_index"]) == 16
        assert len(ratiostats["source_index"]) == 16

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
        md._item_indices = np.arange(100)

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

    def test_structure_progress_callback(self, mock_ds):
        """Test that _structure calls progress_callback with correct values"""
        from unittest.mock import Mock

        md = Metadata(mock_ds)
        callback = Mock()
        md._structure(progress_callback=callback)

        # Verify callback was called for each datum
        assert callback.call_count == len(mock_ds)
        # Check that the last call has the correct final values
        callback.assert_called_with(len(mock_ds) - 1, total=len(mock_ds))

    def test_bin_progress_callback(self, RNG: np.random.Generator):
        """Test that _bin calls progress_callback with correct values"""
        from unittest.mock import Mock

        md = Metadata(None)  # type: ignore
        md_dict = {
            "cat_str": RNG.choice(["A", "B"], size=100).tolist(),
            "con_flt": RNG.random(size=100).tolist(),
            "dis_int": np.arange(100).tolist(),
        }
        md._dataframe = pl.from_dict(md_dict)
        md._factors = dict.fromkeys(md_dict, None)
        md._is_structured = True
        md._item_indices = np.arange(100)

        callback = Mock()
        md._bin(progress_callback=callback)

        # Verify callback was called for each factor
        expected_calls = len(md_dict)
        assert callback.call_count == expected_calls
        # Check that the last call has the correct final values
        callback.assert_called_with(expected_calls, total=expected_calls)

    def test_multidimensional_factors_skipped(self, RNG: np.random.Generator):
        """Test that multi-dimensional factors are skipped during binning and filtered from outputs."""
        md = Metadata(None)  # type: ignore

        # Create a mix of 1D and 2D factors
        md_dict = {
            "factor_1d": RNG.random(size=50).tolist(),
            "embedding_2d": RNG.random(size=(50, 10)).tolist(),  # 2D factor (e.g., embedding)
            "another_1d": RNG.choice(["A", "B", "C"], size=50).tolist(),
        }

        md._dataframe = pl.from_dict(md_dict)
        md._factors = dict.fromkeys(md_dict, None)
        md._target_factors = set(md_dict)
        md._image_factors = set()
        md._is_structured = True
        md._item_indices = np.arange(50)
        md._class_labels = RNG.integers(0, 3, size=50)

        # Trigger factor filtering
        md._build_factors()

        # Trigger binning
        md._bin()

        # Verify that only 1D factors are in factor_names
        assert set(md.factor_names) == {"factor_1d", "another_1d"}
        assert "embedding_2d" not in md.factor_names

        # Verify that only 1D factors are in factor_info
        assert set(md.factor_info.keys()) == {"factor_1d", "another_1d"}
        assert "embedding_2d" not in md.factor_info

        # Verify that factor_data only includes 1D factors
        factor_data = md.factor_data
        assert factor_data.shape == (50, 2)  # 50 samples, 2 1D factors

        # Verify that binned_data only includes 1D factors
        binned_data = md.binned_data
        assert binned_data.shape == (50, 2)  # 50 samples, 2 1D factors

        # Verify that the 2D factor is still in the dataframe (not removed, just skipped)
        assert "embedding_2d" in md.dataframe.columns

        # Verify that _factors has None for the 2D factor
        assert md._factors["factor_1d"] is not None
        assert md._factors["another_1d"] is not None

    def test_target_factors_only_toggle(self, get_mock_od_dataset, RNG: np.random.Generator):
        """Test that toggling target_factors_only properly resets binned data dimensions."""
        # Create an OD dataset with both image-level and target-level factors
        images = [np.random.random((3, 64, 64)) for _ in range(10)]
        labels = [[0, 1] for _ in range(10)]  # 2 targets per image
        bboxes = [[[0, 0, 10, 10], [20, 20, 30, 30]] for _ in range(10)]

        # Create metadata with both image-level and target-level factors
        metadata = []
        for i in range(10):
            metadata.append(
                {
                    "image_factor": f"img_{i}",
                    "shared_factor": i,
                    "target_factor": [f"tgt_{i}_0", f"tgt_{i}_1"],  # 2 target-level values
                }
            )

        ds = get_mock_od_dataset(images, labels, bboxes, metadata=metadata)
        md = Metadata(ds)

        # Initially, should have both image-level and target-level factors
        initial_factor_names = set(md.factor_names)
        initial_binned_shape = md.binned_data.shape

        # Should have at least 2 factors (image_factor and target_factor)
        assert "image_factor" in initial_factor_names
        assert "target_factor" in initial_factor_names
        assert initial_binned_shape[0] == 20  # 10 images * 2 targets each
        assert initial_binned_shape[1] >= 2  # At least 2 factors

        # Set target_factors_only to True - should only have target-level factors
        md.target_factors_only = True
        target_only_factor_names = set(md.factor_names)
        target_only_binned_shape = md.binned_data.shape

        # Should only have target_factor now
        assert "image_factor" not in target_only_factor_names
        assert "target_factor" in target_only_factor_names
        assert target_only_binned_shape[0] == 20  # Still 20 targets
        assert target_only_binned_shape[1] < initial_binned_shape[1]  # Fewer factors

        # Set target_factors_only back to False - should have both factors again
        md.target_factors_only = False
        final_factor_names = set(md.factor_names)
        final_binned_shape = md.binned_data.shape

        # Should have both factors again
        assert "image_factor" in final_factor_names
        assert "target_factor" in final_factor_names
        assert final_binned_shape[0] == 20  # Still 20 targets
        assert final_binned_shape[1] == initial_binned_shape[1]  # Same number of factors as initially

        # Verify the dimensions match factor_names count
        assert final_binned_shape[1] == len(final_factor_names)
