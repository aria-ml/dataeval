import logging
import re
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval._metadata import Metadata
from dataeval.utils.data import (
    DatasetSplits,
    TrainValSplit,
    _build_multilabel_matrix,
    _IterativeStratifiedKFold,
    _multilabel_find_best_split,
    _multilabel_make_splits,
    _multilabel_single_split,
    _simplify_type,
    calculate_validation_fraction,
    flatten_metadata,
    merge_metadata,
    single_split,
    split_dataset,
    validate_groupable,
    validate_labels,
    validate_stratifiable,
)


@pytest.fixture(scope="module")
def _base() -> NDArray[np.intp]:
    return np.arange(5, dtype=np.intp)


@pytest.fixture(scope="module")
def labels(_base):
    """Creates 500 labels with values [0-4]."""
    return np.repeat(_base, 20).astype(np.intp)  # ex. [0, 0, 1, 1, ...]


@pytest.fixture(scope="module")
def groups(_base, RNG):
    """Creates 100 labels and corresponding groups."""
    values = np.tile(_base, 20)  # ex. [0, 1, 0, 1, ...]
    return {"Discrete": values, "Binned": RNG.choice(_base, size=100).astype(np.intp)}


def get_metadata(labels, groups=None) -> Metadata:
    metadata = MagicMock(spec=Metadata)
    metadata.class_labels = labels
    if groups is not None:
        metadata.factor_names = ["Discrete", "Binned"]
        metadata.factor_data = np.column_stack(tuple(groups.values()))
    return metadata


def check_sample_leakage(splits: DatasetSplits):
    test_inds = set(splits.test)

    print("\nChecking for Sample Leakage")
    for fold in splits.folds:
        train_inds = set(fold.train)
        val_inds = set(fold.val)
        assert test_inds.isdisjoint(train_inds), "common elements between train and test"
        assert test_inds.isdisjoint(val_inds), "common elements between val and test"
        assert val_inds.isdisjoint(train_inds), "common elements between train and val"

        assert len(test_inds) + len(train_inds) + len(val_inds) == 100


def check_stratification(labels: NDArray[np.intp], splits: DatasetSplits, tolerance: float):
    """Checks that all folds and optional test split have all labels and tolerable label frequencies."""
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    class_freqs = class_counts / np.sum(class_counts)

    def check_tolerance(labels):
        uniques, counts = np.unique(labels, return_counts=True)
        freq = counts / np.sum(counts)

        assert (uniques == unique_labels).all(), "Mismatch between split labels and all labels"
        assert np.allclose(
            freq,
            class_freqs,
            atol=tolerance,  # , atol=1 / len(class_freqs)
        ), "Frequency difference greater than tolerance"

    if splits.test.size > 0:
        check_tolerance(labels[splits.test])

    for fold in splits.folds:
        check_tolerance(labels[fold.train])


def check_group_leakage(splits: DatasetSplits, metadata: dict[str, NDArray], split_on: list[str]):
    relevant_metadata = np.stack([metadata[key] for key in split_on], axis=1)
    _, groups = np.unique(relevant_metadata, axis=0, return_inverse=True)
    test_groups = set(groups[splits.test])
    for fold in splits.folds:
        train_groups = set(groups[fold.train])
        val_groups = set(groups[fold.val])
        assert test_groups.isdisjoint(train_groups), "common groups between train and test"
        assert test_groups.isdisjoint(val_groups), "common groups between val and test"
        assert val_groups.isdisjoint(train_groups), "common groups between train and val"


####################################################################################################
################################### Test Bad Input Arguments #######################################
####################################################################################################


@pytest.mark.required
class TestInputValidation:
    """Tests the boundaries of the inputs to split dataset."""

    def test_not_stratifiable(self):
        """Tests case where lowest label count is less than partitions."""
        with pytest.raises(ValueError, match="Unable to stratify due to label frequency"):
            assert not validate_stratifiable(np.array([0, 1, 1]), 2)

    def test_stratifiable(self):
        """Tests that equal lowest label count and partitions is valid."""
        validate_stratifiable(np.array([0, 0, 1, 1]), 2)
        pass

    def test_continuous_labels(self, labels):
        """Tests that validate labels raises error with continuous labels."""
        cont_labels = labels.astype(np.float64) + np.random.uniform(-1, 1, size=labels.shape)
        error_statement = "Detected continuous labels. Labels must be discrete for proper stratification"
        with pytest.raises(ValueError, match=error_statement):
            validate_labels(cont_labels, total_partitions=2)

    def test_too_many_partitions(self):
        """Tests that an error is raised if there are more partitions than number of labels."""
        error_statement = (
            r"Total number of labels must be greater than the total number of partitions. "
            r"Got 1 labels and 1 total \[train, val, test\] partitions."
        )

        with pytest.raises(ValueError, match=error_statement):
            validate_labels(np.array([0]), total_partitions=1)

    @pytest.mark.parametrize("folds", [-1, 0])
    def test_invalid_folds(self, folds):
        """Tests that negative and 0 folds are invalid."""
        error_msg = f"Number of folds must be greater than or equal to 1, got {folds}"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=folds, test_frac=0.0, val_frac=0.0)

    @pytest.mark.parametrize("fraction", [-0.001, 1.001])
    def test_invalid_val_frac_values(self, fraction):
        """Tests that 1 fold and val frac is given, but raises error due to val fraction out of bounds [0-1]."""
        error_msg = f"val_frac out of bounds. Must be between 0.0 and 1.0, got {fraction}"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=1, test_frac=0.0, val_frac=fraction)

    @pytest.mark.parametrize("fraction", [-0.001, 1.001])
    def test_invalid_test_frac_values(self, fraction):
        """Tests that an error is raised if test fraction is not between 0.0 and 1.0."""
        error_msg = f"test_frac out of bounds. Must be between 0.0 and 1.0, got {fraction}"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=1, test_frac=fraction, val_frac=0.0)

    def test_invalid_multi_fold_and_val_frac(self):
        """Tests that an error is raised when both >1 fold and val fraction are given."""
        error_msg = "Can only specify val_frac when num_folds equals 1"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=5, test_frac=0.0, val_frac=0.2)

    def test_invalid_one_fold_no_val_frac(self):
        """Tests that 1 fold without specifying val fraction raises error."""
        error_msg = "If num_folds equals 1, must assign a value to val_frac"
        with pytest.raises(ValueError, match=error_msg):
            calculate_validation_fraction(num_folds=1, test_frac=0.0, val_frac=0.0)

    @pytest.mark.parametrize(
        ("tfrac", "vfrac", "expected"),
        [(0.0, 0.1, 0.1), (0.0, 1.0, 1.0), (0.5, 0.5, 0.25), (1.0, 0.5, 0.0), (0.4, 0.4, 0.6 * 0.4)],
    )
    def test_one_fold_valid_fractions(self, tfrac, vfrac, expected):
        """Tests that input fractions correctly calculates validation fraction."""
        validation_fraction = calculate_validation_fraction(
            1,
            test_frac=tfrac,
            val_frac=vfrac,
        )
        assert validation_fraction == expected

    @pytest.mark.parametrize(
        ("folds", "tfrac"),
        [(2, 0.1), (2, 1.0), (2, 0.5), (5, 0.5), (6, 0.4)],
    )
    def test_multi_fold_valid_fractions(self, folds, tfrac):
        """Tests validation calculation with multiple folds."""
        expected = (1.0 / folds) * (1.0 - tfrac)
        result = calculate_validation_fraction(folds, test_frac=tfrac, val_frac=0.0)

        assert expected == result


@pytest.mark.required
class TestGroupData:
    def test_too_few_unique_groups(self, groups):
        """Tests that too many folds over small groups of data raises error and is not groupable."""
        values = groups["Discrete"]
        uniques, group_ids = np.unique(values, axis=0, return_inverse=True)

        error_msg = f"Unique groups ({len(uniques)}) must be greater than num partitions ({len(uniques) + 1})."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            validate_groupable(group_ids, len(uniques) + 1)

    def test_single_unique_group(self):
        """Tests that a single unique group is not groupable."""
        error_msg = "Unique groups (1) must be greater than 1."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            validate_groupable(np.ones(shape=(100, 1), dtype=np.intp), 1000)

    def test_no_valid_groups(self, labels, groups):
        metadata = get_metadata(labels, groups)
        keys = ["Foo", "Bar"]
        error_msg = "Unique groups (1) must be greater than 1."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            split_dataset(
                metadata,
                num_folds=1,
                stratify=False,
                split_on=keys,
                val_frac=0.2,
                test_frac=0.5,
            )


@pytest.mark.required
@pytest.mark.parametrize("val_frac", [0.1, 0.7, 0.99])
def test_split_dataset(labels, val_frac) -> None:
    """Tests no sample leakage at varying validation fractions."""
    metadata = get_metadata(labels)
    splits = split_dataset(metadata, val_frac=val_frac)
    check_sample_leakage(splits)


@pytest.mark.required
@pytest.mark.parametrize("num_folds", [1, 5])
@pytest.mark.parametrize("test_frac", [0.0, 0.25])
class TestFunctionalSplits:
    """Tests split dataset for label miscounts, group and sample leakage, and proper stratification."""

    def test_stratification(self, labels, num_folds, test_frac):
        """Tests stratification with no grouping."""
        metadata = get_metadata(labels)
        val_frac = 0.1 if num_folds == 1 else 0.0
        splits = split_dataset(metadata, num_folds=num_folds, stratify=True, val_frac=val_frac, test_frac=test_frac)
        check_sample_leakage(splits)
        check_stratification(labels, splits, tolerance=0.01)

    def test_grouping(self, labels, groups, num_folds, test_frac):
        """Tests grouping with no stratification."""
        metadata = get_metadata(labels, groups)
        val_frac = 0.1 if num_folds == 1 else 0.0
        keys: list[str] = list(groups.keys())
        splits = split_dataset(
            metadata,
            num_folds=num_folds,
            stratify=False,
            split_on=keys,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        check_sample_leakage(splits)
        check_group_leakage(splits, groups, keys)

    def test_grouped_stratification(self, labels, groups, test_frac, num_folds):
        """Tests grouping and stratification."""
        metadata = get_metadata(labels, groups)
        val_frac = 0.1 if num_folds == 1 else 0.0
        keys: list[str] = list(groups.keys())
        splits = split_dataset(
            metadata,
            num_folds=num_folds,
            stratify=True,
            split_on=keys,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        check_sample_leakage(splits)
        check_group_leakage(splits, groups, keys)
        check_stratification(labels, splits, tolerance=0.15)


@pytest.mark.required
class TestUtilsMetadata:
    duplicate_keys = {
        "a": 1,
        "b": {
            "b1": "b1",
            "b2": "b2",
        },
        "c": {
            "d": [
                {"e": 1, "f": 2, "g": 3},
                {"e": 4, "f": 5, "g": 6},
                {"e": 7, "f": 8, "g": 9, "z": 0},
            ],
            "h": [1.1, 1.2, 1.3],
        },
        "d": {
            "d": {"e": 4, "f": 5, "g": 6},
            "h": 1,
        },
    }

    inconsistent_keys = [
        {"a": 1, "b": [1], "c": [1, 2]},
        {"a": 2},
        {"a": 3, "d": [{"e": {"f": [{"g": 1, "h": 2}]}}]},
    ]

    numpy_value = [{"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112], "point": 4}]

    voc_test = [
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000009.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "600", "height": "300", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "cat",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "53", "ymin": "87", "xmax": "471", "ymax": "420"},
                        "difficult": "0",
                    },
                    {
                        "name": "dog",
                        "pose": "Unspecified",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "158", "ymin": "44", "xmax": "289", "ymax": "167"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Right",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "158", "ymin": "44", "xmax": "289", "ymax": "167"},
                        "difficult": "0",
                    },
                ],
            },
        },
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000036.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "500", "height": "375", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "bicycle",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "120", "ymin": "1", "xmax": "203", "ymax": "35"},
                        "difficult": "0",
                    },
                    {
                        "name": "bicycle",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "117", "ymin": "38", "xmax": "273", "ymax": "121"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Left",
                        "truncated": "0",
                        "occluded": "0",
                        "bndbox": {"xmin": "206", "ymin": "74", "xmax": "395", "ymax": "237"},
                        "difficult": "0",
                        "part": [
                            {"name": "head", "bndbox": {"xmin": "321", "ymin": "75", "xmax": "359", "ymax": "122"}},
                            {"name": "foot", "bndbox": {"xmin": "205", "ymin": "183", "xmax": "240", "ymax": "222"}},
                            {"name": "foot", "bndbox": {"xmin": "209", "ymin": "208", "xmax": "250", "ymax": "237"}},
                            {"name": "hand", "bndbox": {"xmin": "371", "ymin": "204", "xmax": "396", "ymax": "219"}},
                        ],
                    },
                    {
                        "name": "boat",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "24", "ymin": "2", "xmax": "500", "ymax": "188"},
                        "difficult": "0",
                    },
                    {
                        "name": "boat",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "1", "ymin": "187", "xmax": "500", "ymax": "282"},
                        "difficult": "0",
                    },
                ],
            },
        },
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000128.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "500", "height": "375", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "sofa",
                        "pose": "Left",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "11", "ymin": "29", "xmax": "500", "ymax": "375"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Unspecified",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "1", "ymin": "85", "xmax": "361", "ymax": "375"},
                        "difficult": "0",
                        "part": [
                            {"name": "head", "bndbox": {"xmin": "243", "ymin": "88", "xmax": "358", "ymax": "225"}},
                            {"name": "hand", "bndbox": {"xmin": "168", "ymin": "209", "xmax": "216", "ymax": "257"}},
                            {"name": "hand", "bndbox": {"xmin": "94", "ymin": "252", "xmax": "128", "ymax": "308"}},
                        ],
                    },
                    {
                        "name": "person",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "92", "ymin": "173", "xmax": "212", "ymax": "357"},
                        "difficult": "0",
                    },
                ],
            },
        },
    ]

    def test_ignore_lists(self):
        a, d = merge_metadata([self.duplicate_keys], return_dropped=True, ignore_lists=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1],
            "b1": ["b1"],
            "b2": ["b2"],
            "e": [4],
            "f": [5],
            "g": [6],
            "h": [1],
            "_image_index": [0],
        }
        assert d == {"c_d": ["nested_list"], "c_h": ["nested_list"]}

    def test_fully_qualified_keys(self):
        a, d = merge_metadata([self.duplicate_keys], return_dropped=True, fully_qualified=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 1, 1],
            "b_b1": ["b1", "b1", "b1"],
            "b_b2": ["b2", "b2", "b2"],
            "c_d_e": [1, 4, 7],
            "c_d_f": [2, 5, 8],
            "c_d_g": [3, 6, 9],
            "c_h": [1.1, 1.2, 1.3],
            "d_d_e": [4, 4, 4],
            "d_d_f": [5, 5, 5],
            "d_d_g": [6, 6, 6],
            "d_h": [1, 1, 1],
            "_image_index": [0, 0, 0],
        }
        assert d == {"c_d_z": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_duplicate_keys(self, return_numpy):
        a = merge_metadata([self.duplicate_keys], return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 1, 1],
            "b1": ["b1", "b1", "b1"],
            "b2": ["b2", "b2", "b2"],
            "c_d_e": [1, 4, 7],
            "c_d_f": [2, 5, 8],
            "c_d_g": [3, 6, 9],
            "c_h": [1.1, 1.2, 1.3],
            "d_d_e": [4, 4, 4],
            "d_d_f": [5, 5, 5],
            "d_d_g": [6, 6, 6],
            "d_h": [1, 1, 1],
            "_image_index": [0, 0, 0],
        }

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_inconsistent_keys(self, return_numpy):
        a, d = merge_metadata(self.inconsistent_keys, return_dropped=True, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 2, 3],
            "_image_index": [0, 1, 2],
        }
        assert d == {"b": ["inconsistent_key"], "c": ["inconsistent_size"], "d_e_f": ["nested_list"]}

    def test_inconsistent_key(self):
        list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
        reorganized_metadata, dropped_keys = merge_metadata(list_metadata, return_dropped=True)
        assert reorganized_metadata == {
            "common": [1, 1],
            "a": [1, 2],
            "b": [3, 4],
            "source": ["example", "example"],
            "_image_index": [0, 0],
        }
        assert dropped_keys == {"target_c": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_voc_test(self, return_numpy):
        a = merge_metadata(self.voc_test, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "folder": [
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
            ],
            "filename": [
                "2008_000009.jpg",
                "2008_000009.jpg",
                "2008_000009.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000128.jpg",
                "2008_000128.jpg",
                "2008_000128.jpg",
            ],
            "database": [
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
            ],
            "annotation": [
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
            ],
            "image": [
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
            ],
            "width": [600, 600, 600, 500, 500, 500, 500, 500, 500, 500, 500],
            "height": [300, 300, 300, 375, 375, 375, 375, 375, 375, 375, 375],
            "depth": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "segmented": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "name": [
                "cat",
                "dog",
                "person",
                "bicycle",
                "bicycle",
                "person",
                "boat",
                "boat",
                "sofa",
                "person",
                "person",
            ],
            "pose": [
                "Unspecified",
                "Unspecified",
                "Right",
                "Left",
                "Left",
                "Left",
                "Left",
                "Left",
                "Left",
                "Unspecified",
                "Unspecified",
            ],
            "truncated": [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            "occluded": [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            "xmin": [53, 158, 158, 120, 117, 206, 24, 1, 11, 1, 92],
            "ymin": [87, 44, 44, 1, 38, 74, 2, 187, 29, 85, 173],
            "xmax": [471, 289, 289, 203, 273, 395, 500, 500, 500, 361, 212],
            "ymax": [420, 167, 167, 35, 121, 237, 188, 282, 375, 375, 357],
            "difficult": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "_image_index": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        }

    @pytest.mark.filterwarnings("error")
    def test_flatten_metadata_no_dropped_no_warn(self):
        flatten_metadata({"a": {"b": 1, "c": 2}}, return_dropped=False)

    def test_flatten_metadata_no_dropped_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            flatten_metadata(self.inconsistent_keys[0], return_dropped=False)
        assert "Metadata entries were dropped" in caplog.text

    @pytest.mark.filterwarnings("error")
    def test_merge_metadata_no_dropped_no_warn(self):
        merge_metadata([{"a": {"b": 1, "c": 2}}], return_dropped=False)

    def test_merge_metadata_no_dropped_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            merge_metadata(self.inconsistent_keys, return_dropped=False)
        assert "Metadata entries were dropped" in caplog.text

    def test_handle_numpy(self):
        output, dropped = merge_metadata(self.numpy_value, return_dropped=True)
        assert output == {
            "time": [1.2, 3.4, 5.6],
            "altitude": [235, 6789, 101112],
            "point": [4, 4, 4],
            "_image_index": [0, 0, 0],
        }
        assert dropped == {}

    def test_targets_per_image_mismatch(self):
        targets_per_image = [1]
        with pytest.raises(ValueError, match="Number of targets per image must be equal"):
            merge_metadata([{"a": 1}, {"a": 2}], targets_per_image=targets_per_image)

    def test_image_index_key_exists_in_output(self):
        merge_metadatad = merge_metadata([{"a": {"b": 1, "c": 2, "foo": 0}}], image_index_key="foo")
        assert merge_metadatad["foo"] == [0]

    def test_merge_metadata_drop_no_targets(self):
        merge_metadatad = merge_metadata([{"a": 1}, {"a": 2}, {"a": 3}], targets_per_image=[1, 0, 1])
        assert merge_metadatad["a"] == [1, 3]


@pytest.mark.required
class TestCastSimplify:
    @pytest.mark.parametrize(
        ("value", "output"),
        [
            ("123", 123),
            ("12.3", 12.3),
            ("foo", "foo"),
            ([123, "12.3"], [123.0, 12.3]),
            ([123, "foo"], ["123", "foo"]),
            (["123", "456"], [123, 456]),
        ],
    )
    def test_convert_type(self, value, output):
        assert output == _simplify_type(value)


####################################################################################################
################################## Regression / OD Split Tests #####################################
####################################################################################################


def _make_od_metadata(
    n_images: int,
    n_classes: int,
    detections_per_image: int = 3,
    seed: int = 42,
) -> MagicMock:
    """Create a mock Metadata object simulating an OD dataset."""
    rng = np.random.default_rng(seed)
    n_detections = n_images * detections_per_image
    item_indices = np.repeat(np.arange(n_images, dtype=np.intp), detections_per_image)
    class_labels = rng.integers(0, n_classes, size=n_detections).astype(np.intp)

    metadata = MagicMock(spec=Metadata)
    metadata.class_labels = class_labels
    metadata.item_indices = item_indices
    metadata.item_count = n_images
    return metadata


class TestMaxFoldsRegression:
    """Regression tests for the unique_groups=2 bug that capped max_folds at 2."""

    def test_single_split_respects_split_frac(self):
        """single_split should produce splits close to the requested fraction, not 50/50."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 40)  # 200 labels, 5 classes
        index = np.arange(len(labels))
        split = single_split(index, labels, split_frac=0.15, stratified=True)
        ratio = len(split.val) / len(index)
        # Should be within 5% of target, NOT the old 50%
        assert abs(ratio - 0.15) < 0.05, f"Split ratio {ratio:.3f} too far from target 0.15"

    def test_single_split_small_frac(self):
        """10% split should not produce 50/50."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 40)
        index = np.arange(len(labels))
        split = single_split(index, labels, split_frac=0.10, stratified=True)
        ratio = len(split.val) / len(index)
        assert abs(ratio - 0.10) < 0.05

    def test_single_split_large_frac(self):
        """30% split should not produce 50/50."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 40)
        index = np.arange(len(labels))
        split = single_split(index, labels, split_frac=0.30, stratified=True)
        ratio = len(split.val) / len(index)
        assert abs(ratio - 0.30) < 0.05

    def test_split_dataset_proportions_with_test_frac(self):
        """split_dataset with test_frac=0.15 should not put all data in test."""
        labels = np.repeat(np.arange(5, dtype=np.intp), 20)  # 100 labels
        metadata = get_metadata(labels)
        splits = split_dataset(metadata, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        n = len(labels)

        # Test set should be ~15%, NOT 100%
        test_frac_actual = len(splits.test) / n
        assert test_frac_actual < 0.25, f"Test fraction {test_frac_actual:.2f} is too large"
        assert test_frac_actual > 0.05, f"Test fraction {test_frac_actual:.2f} is too small"

        # Train and val should be non-empty
        assert len(splits.folds[0].train) > 0, "Train set is empty"
        assert len(splits.folds[0].val) > 0, "Val set is empty"


class TestBuildMultilabelMatrix:
    """Tests for _build_multilabel_matrix."""

    def test_basic(self):
        class_labels = np.array([0, 1, 2, 0, 1], dtype=np.intp)
        item_indices = np.array([0, 0, 0, 1, 1], dtype=np.intp)
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=2)
        expected = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int8)
        np.testing.assert_array_equal(matrix, expected)

    def test_single_detection_per_image(self):
        class_labels = np.array([0, 1, 2], dtype=np.intp)
        item_indices = np.array([0, 1, 2], dtype=np.intp)
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=3)
        expected = np.eye(3, dtype=np.int8)
        np.testing.assert_array_equal(matrix, expected)

    def test_image_with_no_detections(self):
        """Images with no detections should have all-zero rows."""
        class_labels = np.array([0, 1], dtype=np.intp)
        item_indices = np.array([0, 2], dtype=np.intp)  # image 1 has no detections
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=3)
        np.testing.assert_array_equal(matrix[1], [0, 0])

    def test_duplicate_class_in_image(self):
        """Multiple detections of same class in one image should still be 1."""
        class_labels = np.array([0, 0, 0], dtype=np.intp)
        item_indices = np.array([0, 0, 0], dtype=np.intp)
        matrix = _build_multilabel_matrix(class_labels, item_indices, n_images=1)
        np.testing.assert_array_equal(matrix, [[1]])


class TestIterativeStratifiedKFold:
    """Tests for _IterativeStratifiedKFold."""

    def test_fold_sizes_balanced(self):
        """All folds should have roughly equal sizes."""
        rng = np.random.default_rng(0)
        n, c = 100, 5
        y = rng.integers(0, 2, size=(n, c)).astype(np.int8)
        splitter = _IterativeStratifiedKFold(n_splits=5)
        fold_sizes = [len(test) for _, test in splitter.split(np.arange(n), y)]
        assert sum(fold_sizes) == n
        assert max(fold_sizes) - min(fold_sizes) <= 2, f"Fold sizes too unbalanced: {fold_sizes}"

    def test_no_overlap_between_folds(self):
        """Each sample should appear in exactly one fold's test set."""
        rng = np.random.default_rng(1)
        n, c = 50, 3
        y = rng.integers(0, 2, size=(n, c)).astype(np.int8)
        splitter = _IterativeStratifiedKFold(n_splits=3)
        all_test = []
        for _, test in splitter.split(np.arange(n), y):
            all_test.extend(test.tolist())
        assert sorted(all_test) == list(range(n))

    def test_train_test_complementary(self):
        """Train and test indices should be complementary for each fold."""
        n = 30
        y = np.eye(5, dtype=np.int8)[:3].T  # 5 samples, 3 classes
        y = np.tile(y, (6, 1))  # 30 samples
        splitter = _IterativeStratifiedKFold(n_splits=3)
        for train, test in splitter.split(np.arange(n), y):
            assert len(train) + len(test) == n
            assert set(train.tolist()).isdisjoint(set(test.tolist()))

    def test_rare_labels_distributed(self):
        """Rare labels should appear in multiple folds, not all in one."""
        n = 100
        y = np.zeros((n, 4), dtype=np.int8)
        y[:, 0] = 1  # common label: all 100 images
        y[:4, 1] = 1  # rare label: only 4 images
        y[:50, 2] = 1  # medium label
        y[:20, 3] = 1  # less common
        splitter = _IterativeStratifiedKFold(n_splits=4)
        for _, test in splitter.split(np.arange(n), y):
            # Rare label (4 total) should have exactly 1 per fold
            assert y[test, 1].sum() >= 1


class TestMultilabelSplitFunctions:
    """Tests for _multilabel_single_split, _multilabel_make_splits, _multilabel_find_best_split."""

    @pytest.fixture
    def multilabel_data(self):
        rng = np.random.default_rng(42)
        n_images, n_classes = 200, 8
        ml = rng.integers(0, 2, size=(n_images, n_classes)).astype(np.int8)
        # Ensure every class appears in at least 10 images
        for c in range(n_classes):
            if ml[:, c].sum() < 10:
                ml[rng.choice(n_images, 10, replace=False), c] = 1
        return ml

    def test_single_split_no_leakage(self, multilabel_data):
        ml = multilabel_data
        index = np.arange(len(ml))
        split = _multilabel_single_split(index, ml, split_frac=0.2)
        assert set(split.train.tolist()).isdisjoint(set(split.val.tolist()))
        assert len(split.train) + len(split.val) == len(ml)

    def test_single_split_respects_fraction(self, multilabel_data):
        ml = multilabel_data
        index = np.arange(len(ml))
        split = _multilabel_single_split(index, ml, split_frac=0.2)
        ratio = len(split.val) / len(index)
        assert abs(ratio - 0.2) < 0.05

    def test_make_splits_covers_all_indices(self, multilabel_data):
        ml = multilabel_data
        index = np.arange(len(ml))
        splits = _multilabel_make_splits(index, ml, n_folds=5)
        assert len(splits) == 5
        for split in splits:
            assert len(split.train) + len(split.val) == len(ml)

    def test_find_best_split_prefers_balanced(self):
        """find_best_split should prefer splits with balanced per-label frequencies."""
        n = 100
        ml = np.zeros((n, 2), dtype=np.int8)
        ml[:50, 0] = 1  # class 0: first 50
        ml[50:, 1] = 1  # class 1: last 50

        good_split = TrainValSplit(np.arange(0, 80, dtype=np.intp), np.arange(80, 100, dtype=np.intp))
        bad_split = TrainValSplit(np.arange(50, 100, dtype=np.intp), np.arange(0, 50, dtype=np.intp))
        best = _multilabel_find_best_split(ml, [good_split, bad_split], 0.2)
        # good_split has both classes in train; bad_split puts all of class 0 in val
        assert np.array_equal(best.train, good_split.train)


class TestODSplitDataset:
    """Integration tests for split_dataset with OD (multi-label) datasets."""

    def test_od_returns_image_level_indices(self):
        """All returned indices must be valid image indices, not detection indices."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        n_images = 200
        all_indices = np.concatenate([splits.test, splits.folds[0].train, splits.folds[0].val])
        assert all_indices.max() < n_images, "Index exceeds number of images"
        assert all_indices.min() >= 0

    def test_od_no_leakage(self):
        """No image should appear in more than one split."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        train = set(splits.folds[0].train.tolist())
        val = set(splits.folds[0].val.tolist())
        test = set(splits.test.tolist())
        assert train.isdisjoint(val), "Train/Val overlap"
        assert train.isdisjoint(test), "Train/Test overlap"
        assert val.isdisjoint(test), "Val/Test overlap"
        assert len(train | val | test) == 200

    def test_od_split_proportions(self):
        """OD split proportions should approximate requested fractions."""
        n_images = 500
        meta = _make_od_metadata(n_images=n_images, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15)
        test_frac = len(splits.test) / n_images
        train_frac = len(splits.folds[0].train) / n_images
        val_frac = len(splits.folds[0].val) / n_images
        assert abs(test_frac - 0.15) < 0.05, f"Test frac {test_frac:.3f} not ~0.15"
        assert train_frac > 0.5, f"Train frac {train_frac:.3f} too small"
        assert val_frac > 0.05, f"Val frac {val_frac:.3f} too small"

    def test_od_no_stratify(self):
        """OD split without stratification should still use image-level indices."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=1, stratify=False, test_frac=0.15, val_frac=0.15)
        all_indices = np.concatenate([splits.test, splits.folds[0].train, splits.folds[0].val])
        assert all_indices.max() < 200
        assert len(set(all_indices.tolist())) == 200

    def test_od_multi_fold(self):
        """OD split with multiple folds should work correctly."""
        meta = _make_od_metadata(n_images=200, n_classes=10)
        splits = split_dataset(meta, num_folds=3, stratify=True, test_frac=0.15)
        assert len(splits.folds) == 3
        for fold in splits.folds:
            all_idx = np.concatenate([splits.test, fold.train, fold.val])
            assert len(set(all_idx.tolist())) == 200

    def test_od_class_representation_in_splits(self):
        """Every class present in the dataset should appear in the test split."""
        meta = _make_od_metadata(n_images=500, n_classes=10, detections_per_image=5, seed=0)
        splits = split_dataset(meta, num_folds=1, stratify=True, test_frac=0.2, val_frac=0.2)
        ml = _build_multilabel_matrix(meta.class_labels, meta.item_indices, 500)
        test_classes = ml[splits.test].sum(axis=0)
        # All classes should appear in the test split
        assert (test_classes > 0).all(), f"Missing classes in test: {test_classes}"

    def test_od_split_on_warns(self, caplog):
        """split_on should emit a warning for OD datasets."""
        meta = _make_od_metadata(n_images=100, n_classes=5)
        with caplog.at_level(logging.WARNING):
            split_dataset(meta, num_folds=1, stratify=True, test_frac=0.15, val_frac=0.15, split_on=["foo"])
        assert "split_on is not supported for object detection" in caplog.text

    def test_ic_dataset_unchanged(self, labels):
        """IC datasets should still use the standard single-label path."""
        metadata = get_metadata(labels)
        splits = split_dataset(metadata, num_folds=1, stratify=True, val_frac=0.2, test_frac=0.2)
        check_sample_leakage(splits)
        check_stratification(labels, splits, tolerance=0.02)
