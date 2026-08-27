import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._columns import binned
from dataeval.core import compute_stats
from dataeval.core._compute_ratios import compute_ratios
from dataeval.core._label_stats import label_stats
from dataeval.data import unzip_dataset
from dataeval.exceptions import ShapeMismatchError
from dataeval.flags import ImageStats
from dataeval.types import FactorInfo
from tests.embeddings.test_embeddings import MockDataset


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
    """Test collate aggregates MAITE style data into separate collections from tuple return."""

    @pytest.mark.parametrize(
        ("data", "labels", "metadata", "factors"),
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
        """Tests common (input, target, metadata) dataset output."""
        ds = MockDataset(data, labels, metadata)
        md = Metadata(ds)

        # Ten values sit at the unit level; twenty, one per detection, sit at the instance level.
        md.add_factors({"a": np.random.random((factors,))}, level="unit" if factors == 10 else "instance")
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

        # Use label_stats directly with flat class_labels and item_indices
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)

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

        # Use label_stats directly with flat class_labels and item_indices
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)

        assert stats["label_counts_per_image"] == [2, 0, 2, 2, 2, 0, 2, 2, 2, 2]

        _img, _box = unzip_dataset(mock_ds, False)
        imgstats = compute_stats(
            _img,
            boxes=_box,
            stats=ImageStats.PIXEL | ImageStats.VISUAL,
            per_image=True,
            per_target=False,
        )
        _img, _box = unzip_dataset(mock_ds, True)
        boxstats = compute_stats(
            _img,
            boxes=_box,
            stats=ImageStats.PIXEL | ImageStats.VISUAL,
            per_image=False,
            per_target=True,
        )
        ratiostats = compute_ratios(imgstats, target_stats_output=boxstats)
        assert len(imgstats["source_index"]) == 10
        assert len(boxstats["source_index"]) == 16
        assert len(ratiostats["source_index"]) == 16

    def test_add_factors_preserves_existing_factor_info(self):
        """Regression: add_factors after factor_info should not drop existing factors."""
        from tests.conftest import to_metadata

        md = to_metadata(
            {"brightness": np.random.rand(50).tolist()},
            list(range(50)),
            {"brightness": 5},
        )

        # Step 1: Access factor_info to trigger binning
        info1 = md.factor_info
        original_factors = set(info1.keys())
        assert "brightness" in original_factors

        # Step 2: Add a new factor
        md.add_factors({"contrast": np.random.rand(50)}, level="unit")

        # Step 3: Access factor_info again - all original factors should still be present
        info2 = md.factor_info
        assert original_factors.issubset(set(info2.keys())), (
            f"Original factors {original_factors} were dropped after add_factors. "
            f"Remaining factors: {set(info2.keys())}"
        )
        assert "contrast" in info2

    def test_mismatch_factor_length(self, mock_metadata):
        with pytest.raises(ShapeMismatchError, match="must have length"):
            mock_metadata.add_factors({"a": np.random.random((20,))}, level="unit")

    def test_add_empty_factors(self):
        md = Metadata(None)  # type: ignore
        md._factors = set()
        md._count = 0
        md._is_structured = True
        md.add_factors({})
        assert md.factor_names == []

    def test_all_factor_types(self, RNG: np.random.Generator):
        md = Metadata.from_factors(
            {
                "cat_str": RNG.choice(["A", "B"], size=100),
                "con_flt": RNG.random(size=100),
                "dis_flt": RNG.choice([0.1, 0.2, 0.4, 0.6, 0.8], size=100),
                "dis_int": np.arange(100),
            },
        )

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

    def test_include_single_string(self):
        """Regression: setting include to a bare string should treat it as one factor name."""
        md = Metadata(None, include="height")  # type: ignore
        assert md._include == {"height"}

        md.include = "width"
        assert md._include == {"width"}

    def test_exclude_single_string(self):
        """Regression: setting exclude to a bare string should treat it as one factor name."""
        md = Metadata(None, exclude="height")  # type: ignore
        assert md._exclude == {"height"}

        md.exclude = "width"
        assert md._exclude == {"width"}

    def test_dropped_factors(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._dropped_factors = {"b": ["foo"], "c": ["bar"]}
        assert md.dropped_factors == {"b": ["foo"], "c": ["bar"]}

    def test_unknown_target(self):
        # MAITE-shape validation now rejects unsupported target types at construction
        # rather than waiting for _structure(); the error is a MaiteShapeError (TypeError).
        with pytest.raises(TypeError, match=r"dataset\[0\]\[1\]"):
            Metadata([(np.zeros((3, 16, 16)), "THIS IS NOT A TARGET", {"id": 0})])  # type: ignore

    def test_mixed_target(self):
        md = Metadata(
            [
                (np.zeros((3, 16, 16)), np.zeros((3,)), {"id": 0}),
                (np.zeros((3, 16, 16)), ObjectDetectionTarget([[0, 0, 0, 0]], [0], [0, 0, 0]), {"id": 0}),
            ],  # type: ignore
        )
        # The first datum's target selects the strategy; the second one then fails
        # to satisfy it.
        with pytest.raises(TypeError, match="Encountered unsupported target type"):
            md._structure()

    def test_process_include(self, mock_ds):
        md = Metadata(mock_ds, include=["id"])
        md._bin()

    def test_process_exclude(self, mock_ds):
        md = Metadata(mock_ds, exclude=["id"])
        md._bin()

    def test_contiguous_factor_bins_setter(self):
        md = Metadata(None)  # type: ignore
        md._is_binned = True
        md.continuous_factor_bins = {"a": 10}
        assert not md._is_binned
        assert md._continuous_factor_bins == {"a": 10}

    def test_contiguous_factor_bins_setter_no_op(self):
        md = Metadata(None, continuous_factor_bins={"a": 10})  # type: ignore
        md._is_binned = True
        md.continuous_factor_bins = {"a": 10}
        assert md._is_binned
        assert md._continuous_factor_bins == {"a": 10}

    def test_auto_bin_method_setter(self):
        md = Metadata(None)  # type: ignore
        md._is_binned = True
        md.auto_bin_method = "clusters"
        assert not md._is_binned
        assert md._auto_bin_method == "clusters"

    def test_auto_bin_method_setter_no_op(self):
        md = Metadata(None, auto_bin_method="clusters")  # type: ignore
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
        md._factors = {"foo"}
        md._exclude = {"foo"}
        assert md.factor_data.size == 0

    def test_empty_factor_data(self):
        md = Metadata(None)  # type: ignore
        md._is_structured = True
        md._factors = {"foo"}
        md._exclude = {"foo"}
        assert md.factor_data.size == 0

    @pytest.mark.parametrize(
        ("is_binned", "exists"),
        [
            (True, True),
            (True, False),
            (False, False),
        ],
    )
    def test_reset_bins(self, is_binned, exists):
        """The companion column and its cached info go together, whatever _is_binned says."""
        col = "foo"
        col_bn = binned(col)
        md = Metadata.from_factors({col: np.array([0])})
        if exists:
            md._store = md._store.with_column("unit", pl.Series(col_bn, [0]))
        md._factor_cache = {col: FactorInfo("continuous", is_binned=exists)}
        md._is_binned = is_binned
        md._reset_bins()
        assert not md._is_binned
        assert col_bn not in md._store.columns
        # Info survives only where there was no column to drop; the factor stays visible
        # either way, since _reset_bins clears binning and not the factor registry.
        assert (col in md._factor_cache) is not exists
        assert md._factors == {col}

    def test_structure_progress_callback(self, mock_ds):
        """Test that _structure calls progress_callback with correct values."""
        from unittest.mock import Mock

        md = Metadata(mock_ds)
        callback = Mock()
        md._structure(progress_callback=callback)

        # Verify callback was called for each datum
        assert callback.call_count == len(mock_ds)
        # Progress counts items completed, so the last call reports the total.
        callback.assert_called_with(len(mock_ds), total=len(mock_ds))

    def test_bin_progress_callback(self, RNG: np.random.Generator):
        """Test that _bin calls progress_callback with correct values."""
        from unittest.mock import Mock

        md_dict = {
            "cat_str": RNG.choice(["A", "B"], size=100),
            "con_flt": RNG.random(size=100),
            "dis_int": np.arange(100),
        }
        md = Metadata.from_factors(md_dict)

        callback = Mock()
        md._bin(progress_callback=callback)

        # Verify callback was called for each factor
        expected_calls = len(md_dict)
        assert callback.call_count == expected_calls
        # Check that the last call has the correct final values
        callback.assert_called_with(expected_calls, total=expected_calls)

    def test_multidimensional_factors_skipped(self, RNG: np.random.Generator):
        """Test that multi-dimensional factors are skipped during binning and filtered from outputs."""
        md = Metadata.from_factors(
            {
                "factor_1d": RNG.random(size=50),
                "another_1d": RNG.choice(["A", "B", "C"], size=50),
            },
            class_labels=RNG.integers(0, 3, size=50),
        )
        # A 2D factor (e.g. an embedding) sitting in the dataframe as a polars List column.
        # add_factors refuses to create one, so it is written directly.
        md._store = md._store.with_column("unit", pl.Series("embedding_2d", RNG.random(size=(50, 10))))
        md._factors_by_level.setdefault("unit", set()).add("embedding_2d")
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
        factor_data = md.rows_at(md.view).select(md.factor_names).to_numpy()
        assert factor_data.shape == (50, 2)  # 50 samples, 2 1D factors

        # Verify that binned_data only includes 1D factors
        binned_data = md.factor_data
        assert binned_data.shape == (50, 2)  # 50 samples, 2 1D factors

        # Verify that the 2D factor is still in the dataframe (not removed, just skipped)
        assert "embedding_2d" in md.dataframe.columns

        # The 1D factors were processed; the 2D one never entered the visible set.
        assert md._factor_cache["factor_1d"] is not None
        assert md._factor_cache["another_1d"] is not None
        assert "embedding_2d" not in md._factors

    def test_add_factors_skips_multidimensional(self, RNG: np.random.Generator):
        """A multi-dimensional array has no single-column form, so add_factors drops it."""
        md = Metadata.from_factors(
            {"factor_1d": RNG.random(size=50)},
            class_labels=RNG.integers(0, 3, size=50),
        )

        md.add_factors({"embedding_2d": RNG.random(size=(50, 10)), "scalar": RNG.random(size=50)}, level="unit")

        assert "embedding_2d" not in md.dataframe.columns
        assert md.dropped_factors["embedding_2d"] == ["multi_dimensional"]
        assert "scalar" in md.factor_names

    def test_inherited_toggle(self, get_mock_od_dataset, RNG: np.random.Generator):
        """Test that toggling inherited properly resets binned data dimensions."""
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
                },
            )

        ds = get_mock_od_dataset(images, labels, bboxes, metadata=metadata)
        md = Metadata(ds)

        # Initially, should have both image-level and target-level factors
        initial_factor_names = set(md.factor_names)
        initial_binned_shape = md.factor_data.shape

        # Should have at least 2 factors (image_factor and target_factor)
        assert "image_factor" in initial_factor_names
        assert "target_factor" in initial_factor_names
        assert initial_binned_shape[0] == 20  # 10 images * 2 targets each
        assert initial_binned_shape[1] >= 2  # At least 2 factors

        # Drop inherited factors - should only have instance-native factors
        md.inherited = False
        target_only_factor_names = set(md.factor_names)
        target_only_binned_shape = md.factor_data.shape

        # Should only have target_factor now
        assert "image_factor" not in target_only_factor_names
        assert "target_factor" in target_only_factor_names
        assert target_only_binned_shape[0] == 20  # Still 20 targets
        assert target_only_binned_shape[1] < initial_binned_shape[1]  # Fewer factors

        # Restore inherited factors - should have both factors again
        md.inherited = True
        final_factor_names = set(md.factor_names)
        final_binned_shape = md.factor_data.shape

        # Should have both factors again
        assert "image_factor" in final_factor_names
        assert "target_factor" in final_factor_names
        assert final_binned_shape[0] == 20  # Still 20 targets
        assert final_binned_shape[1] == initial_binned_shape[1]  # Same number of factors as initially

        # Verify the dimensions match factor_names count
        assert final_binned_shape[1] == len(final_factor_names)

    def test_add_factors_without_a_destination_fails(self, get_od_dataset):
        """Inference from array length was removed; the destination has to be stated."""
        images = np.random.random((5, 3, 16, 16))
        md = Metadata(get_od_dataset(images, 2, True))
        md._structure()

        with pytest.raises(ValueError, match="destination"):
            md.add_factors({"a": [1, 2, 3]})

    def test_add_factors_places_a_mapping_at_one_level(self, get_od_dataset):
        """``level`` applies to the whole mapping, so mixing levels is one call per level."""
        images = np.random.random((5, 3, 16, 16))
        md = Metadata(get_od_dataset(images, 2, True))
        md._structure()

        md.add_factors({"bright": np.arange(5.0)}, level="unit")
        md.add_factors({"iou": np.arange(10.0)}, level="instance")

        assert md.factor_info["bright"].level == "unit"
        assert md.factor_info["iou"].level == "instance"

    def test_add_factors_invalid_level(self, get_od_dataset):
        """An explicit level outside the dataset's schema is rejected."""
        images = np.random.random((5, 3, 16, 16))
        md = Metadata(get_od_dataset(images, 2, True))

        with pytest.raises(ValueError, match="Unknown level 'invalid'"):
            md.add_factors({"a": [1, 2, 3]}, level="invalid")  # type: ignore[arg-type]

    def test_filter_by_factor_with_condition(self, get_od_dataset):
        """Test filter_by_factor returns filtered results (line 1199)."""
        images = np.random.random((5, 3, 16, 16))
        metadata = [{"continuous_val": float(i * 10.0), "categorical_val": f"cat_{i}"} for i in range(5)]

        dataset = get_od_dataset(images, 2, True, metadata=metadata)

        md = Metadata(dataset, continuous_factor_bins={"continuous_val": 3})

        # Filter for only continuous factors
        result = md.filter_by_factor(lambda name, info: info.factor_type == "continuous")
        assert result.shape[0] >= 5  # At least 5 samples (could be more with targets)
        assert result.shape[1] >= 1  # At least 1 continuous factor


@pytest.mark.required
class TestMetadataIndexing:
    """``metadata[...]`` reads the factor matrix by row, by name, or by slice."""

    @staticmethod
    def _metadata():
        from tests.conftest import to_metadata

        return to_metadata({"a": [1.0, 2.0, 3.0, 4.0], "b": [0, 1, 0, 1]}, [0, 1, 0, 1])

    def test_a_name_selects_that_factor_column(self):
        metadata = self._metadata()
        column = metadata["a"]
        assert column.shape == (metadata.factor_data.shape[0],)
        np.testing.assert_array_equal(column, metadata.factor_data[:, metadata.factor_names.index("a")])

    def test_a_slice_selects_rows(self):
        metadata = self._metadata()
        np.testing.assert_array_equal(metadata[1:3], metadata.factor_data[1:3])

    def test_an_unknown_name_is_a_key_error(self):
        with pytest.raises(KeyError, match="not found in metadata"):
            self._metadata()["nope"]

    def test_an_unusable_index_type_is_rejected(self):
        with pytest.raises(TypeError, match="int, str, or slice"):
            self._metadata()[object()]  # type: ignore[index]


@pytest.mark.required
class TestMetadataWithNoFactors:
    """A dataset carrying no factors still answers with correctly-shaped emptiness."""

    @staticmethod
    def _metadata(get_od_dataset):
        # `id` is auto-generated, so excluding it is what leaves no factors at all.
        return Metadata(get_od_dataset(4, metadata=[{}] * 4), exclude=["id"])

    def test_factor_values_is_an_empty_projection(self, get_od_dataset):
        values = self._metadata(get_od_dataset).factor_values
        assert values.shape[1] == 0
        assert values.dtype == np.float64

    def test_item_count_structures_on_demand(self, get_od_dataset):
        """An unfiltered instance has not been structured yet, so the count triggers it."""
        assert self._metadata(get_od_dataset).item_count == 4


@pytest.mark.required
def test_item_count_on_an_empty_dataset_structures_to_find_zero():
    """A zero-length dataset leaves `_count` at 0, which is what triggers the extraction."""

    class _Empty:
        metadata = {"id": "empty"}

        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int):
            raise IndexError(index)

    with pytest.warns(UserWarning, match="empty dataset"):
        metadata = Metadata(_Empty())  # type: ignore[arg-type]  # type: ignore[arg-type]
    assert metadata.item_count == 0


@pytest.mark.required
class TestIdempotentSetters:
    """Assigning the value already in place changes nothing and rebuilds nothing."""

    def test_setting_view_to_the_current_level_is_a_no_op(self, get_od_dataset):
        metadata = Metadata(get_od_dataset(4, targets_per_image=2))
        metadata.view = metadata.view
        before = metadata.dataframe
        metadata.view = metadata.view
        assert metadata.dataframe.equals(before)

    def test_setting_inherited_to_its_current_value_is_a_no_op(self, get_od_dataset):
        metadata = Metadata(get_od_dataset(4, targets_per_image=2))
        before = metadata.inherited
        metadata.inherited = before
        assert metadata.inherited == before


@pytest.mark.required
def test_accept_passes_over_factors_already_ratified(get_od_dataset):
    """Accepting twice is idempotent: the second pass has nothing left to ratify."""
    metadata = Metadata(get_od_dataset(6, metadata=[{"w": "a", "n": float(i)} for i in range(6)]))
    metadata.factor_data  # noqa: B018  # bin, so the factors gain a derived encoding

    names = tuple(metadata.factor_names)
    metadata.accept(*names)
    accepted = {name: metadata._factor_info[name].encoding for name in names}

    # Naming them again reaches the skip: their provenance is no longer "derived".
    metadata.accept(*names)
    assert {name: metadata._factor_info[name].encoding for name in names} == accepted
