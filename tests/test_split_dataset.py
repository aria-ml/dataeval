import numpy as np
import pytest

from src.dataeval._internal.split_dataset import split_dataset

groupnames = [
    None,
    ["DiscreteCorrelated"],
    ["DiscreteUncorrelated"],
    ["ContinuousCorrelated"],
    ["ContinuousUncorrelated"],
    ["AngleCorrelated"],
    ["AngleUncorrelated"],
    ["DiscreteCorrelated", "ContinuousCorrelated"],
    ["AngleCorrelated", "AngleUncorrelated"],
]


def test_missing_val_frac_arg(labels_with_metadata):
    labels, metadata = labels_with_metadata
    with pytest.raises(UnboundLocalError):
        split_dataset(labels)
    with pytest.raises(UnboundLocalError):
        split_dataset(labels, num_folds=1)


def test_invalid_val_frac_arg(labels_with_metadata):
    labels, metadata = labels_with_metadata
    with pytest.raises(ValueError):
        split_dataset(labels, num_folds=2, val_frac=0.25)


def test_missing_metadata(labels_with_metadata):
    labels, metadata = labels_with_metadata
    with pytest.raises(UnboundLocalError):
        split_dataset(labels, num_folds=2, split_on=["DiscreteCorrelated"], metadata=None)
    with pytest.raises(UnboundLocalError):
        split_dataset(labels, num_folds=2, split_on=["DiscreteCorrelated"])


def test_stratification_warning(labels_with_metadata):
    labels, metadata = labels_with_metadata
    with pytest.warns(UserWarning):
        split_dataset(labels, num_folds=labels.shape[0] // 2, stratify=True)


def test_continuous_labels(labels_with_metadata):
    labels, metadata = labels_with_metadata
    labels = labels.astype(float) + np.random.uniform(-1, 1, size=labels.shape)
    with pytest.raises(ValueError):
        split_dataset(labels, num_folds=2)


def test_too_many_partitions(labels_with_metadata):
    labels, metadata = labels_with_metadata
    with pytest.raises(IndexError):
        split_dataset(labels, num_folds=labels.shape[0] + 1)


def test_label_metadata_mismatch(labels_with_metadata):
    labels, metadata = labels_with_metadata
    metadata["DiscreteCorrelated"] = metadata["DiscreteCorrelated"][:-1]
    with pytest.raises(IndexError):
        split_dataset(labels, num_folds=2, split_on=["DiscreteCorrelated"], metadata=metadata)


@pytest.mark.parametrize("num_folds", [5, 10, 15])
@pytest.mark.parametrize("test_frac", [None, 0.25, 0.67])
def test_stratification(labels_with_metadata, num_folds, test_frac):
    labels, metadata = labels_with_metadata
    splits = split_dataset(labels, num_folds=num_folds, stratify=True, metadata=metadata, test_frac=test_frac)
    check_sample_leakage(splits)
    check_stratification(labels, splits)


@pytest.mark.parametrize("num_folds", [5, 10])
@pytest.mark.parametrize("stratify", [False, True])
@pytest.mark.parametrize("split_on", groupnames)
@pytest.mark.parametrize("test_frac", [None, 0.25])
def test_grouping(labels_with_metadata, num_folds, stratify, split_on, test_frac):
    labels, metadata = labels_with_metadata
    splits = split_dataset(
        labels, num_folds=num_folds, stratify=stratify, split_on=split_on, metadata=metadata, test_frac=test_frac
    )
    check_sample_leakage(splits)
    check_group_leakage(splits, metadata, split_on)


@pytest.mark.parametrize("split_on", groupnames)
@pytest.mark.parametrize("test_frac", [None, 0.25, 0.75])
@pytest.mark.parametrize("val_frac", [0.125, 0.5])
def test_single_fold(labels_with_metadata, split_on, test_frac, val_frac):
    labels, metadata = labels_with_metadata
    splits = split_dataset(
        labels, num_folds=1, stratify=True, split_on=split_on, metadata=metadata, test_frac=test_frac, val_frac=val_frac
    )
    check_sample_leakage(splits)


@pytest.mark.parametrize("num_folds", [2, 8, 16])
@pytest.mark.parametrize("split_on", groupnames)
@pytest.mark.parametrize("test_frac", [None, 0.25, 0.75])
def test_multi_fold(labels_with_metadata, split_on, test_frac, num_folds):
    labels, metadata = labels_with_metadata
    splits = split_dataset(
        labels,
        num_folds=num_folds,
        stratify=True,
        split_on=split_on,
        metadata=metadata,
        test_frac=test_frac,
    )
    check_sample_leakage(splits)


def check_sample_leakage(splits):
    test_inds = set(splits.get("test", []))
    print("\nChecking for Sample Leakage")
    for foldname, folddict in splits.items():
        if foldname == "test":
            continue
        train_inds = set(folddict["train"])
        val_inds = set(folddict["val"])
        assert test_inds.isdisjoint(train_inds), "common elements between train and test"
        assert test_inds.isdisjoint(val_inds), "common elements between val and test"
        assert val_inds.isdisjoint(train_inds), "common elements between train and val"


def check_stratification(labels, splits):
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    class_freqs = class_counts / class_counts.sum()
    test_inds = splits.get("test")
    if test_inds is not None:
        test_labels = labels[test_inds]
        unique_test, test_counts = np.unique(test_labels, return_counts=True)
        test_freq = test_counts / test_counts.sum()
        assert len(unique_test) == len(unique_labels), "Test set does not contain all labels"
        assert (unique_test == unique_labels).all(), "Mismatch between test labels and all labels"
        assert np.allclose(
            test_freq,
            class_freqs,
            rtol=0.1,  # , atol=1 / len(class_freqs)
        ), "Test set difference greater than tolerance"
    for foldname, folddict in splits.items():
        if foldname == "test":
            continue
        train_labels = labels[folddict["train"]]
        unique_train, train_counts = np.unique(train_labels, return_counts=True)
        train_freq = train_counts / train_counts.sum()
        assert len(unique_train) == len(unique_labels), "Test set does not contain all labels"
        assert (unique_train == unique_labels).all(), "Mismatch between test labels and all labels"
        assert np.allclose(
            train_freq,
            class_freqs,
            rtol=0.1,  # , atol=1 / len(class_freqs)
        ), "Train set difference greater than tolerance"


def check_group_leakage(splits, metadata, split_on):
    if split_on is None:
        return
    relevant_metadata = np.stack([metadata[key] for key in split_on], axis=1)
    _, groups = np.unique(relevant_metadata, axis=0, return_inverse=True)
    test_inds = splits.get("test", [])
    test_groups = set(groups[test_inds])
    for foldname, folddict in splits.items():
        if foldname == "test":
            continue
        train_groups = set(groups[folddict["train"]])
        val_groups = set(groups[folddict["val"]])
        assert test_groups.isdisjoint(train_groups), "common groups between train and test"
        assert test_groups.isdisjoint(val_groups), "common groups between val and test"
        assert val_groups.isdisjoint(train_groups), "common groups between train and val"


# @pytest.mark.parametrize("nfolds", [1, 3, None])
# @pytest.mark.parametrize("test_frac", [0.25, 0.66, None])
# @pytest.mark.parametrize("val_frac", [0.125, 0.66, None])
# @pytest.mark.parametrize("split_on", groupnames)
# @pytest.mark.parametrize("stratify", [True, False, None])
# class TestDatasetSplitter:
#     def test_invalid_args(self, labels_with_metadata, nfolds, test_frac, val_frac, split_on, stratify):
#         labels, metadata = labels_with_metadata
#         if val_frac and (nfolds not in [None, 1]):
#             with pytest.raises(ValueError):
#                 split_dataset(labels, nfolds, test_frac, val_frac, split_on, metadata, stratify)
#         elif (not val_frac) and (nfolds == 1 or (not nfolds)):
#             with pytest.raises(UnboundLocalError):
#                 split_dataset(labels, nfolds, test_frac, val_frac, split_on, metadata, stratify)
#         elif split_on is not None:
#             with pytest.raises(UnboundLocalError):
#                 split_dataset(labels, 1, test_frac, 0.125, split_on, None, stratify)
#         else:
#             splits = split_dataset(labels, nfolds, test_frac, val_frac, split_on, metadata, stratify)
#             self.check_sample_leakage(splits)
#             if split_on:
#                 self.check_group_leakage(splits, metadata, split_on)
#             if stratify and (not split_on):
#                 self.check_stratification(labels, splits)

#     def check_sample_leakage(self, splits):
#         test_inds = set(splits.get("test", []))
#         print("\nChecking for Sample Leakage")
#         for foldname, folddict in splits.items():
#             if foldname == "test":
#                 continue
#             train_inds = set(folddict["train"])
#             val_inds = set(folddict["val"])
#             assert test_inds.isdisjoint(train_inds), "common elements between train and test"
#             assert test_inds.isdisjoint(val_inds), "common elements between val and test"
#             assert val_inds.isdisjoint(train_inds), "common elements between train and val"
#             print(f"{foldname} ok. No sample leakage detected")

#     def check_stratification(self, labels, splits):
#         unique_labels, class_counts = np.unique(labels, return_counts=True)
#         class_freqs = class_counts / class_counts.sum()
#         test_inds = splits.get("test")
#         print("\nChecking label stratification")
#         if test_inds is not None:
#             test_labels = labels[test_inds]
#             unique_test, test_counts = np.unique(test_labels, return_counts=True)
#             test_freq = test_counts / test_counts.sum()
#             assert len(unique_test) == len(unique_labels), "Test set does not contain all labels"
#             assert (unique_test == unique_labels).all(), "Mismatch between test labels and all labels"
#             assert np.allclose(
#                 test_freq, class_freqs, rtol=0.05, atol=1 / len(class_freqs)
#             ), "Test set difference greater than tolerance"
#             print("Test split ok.")
#         for foldname, folddict in splits.items():
#             if foldname == "test":
#                 continue
#             train_labels = labels[folddict["train"]]
#             unique_train, train_counts = np.unique(train_labels, return_counts=True)
#             train_freq = train_counts / train_counts.sum()
#             assert len(unique_train) == len(unique_labels), "Test set does not contain all labels"
#             assert (unique_train == unique_labels).all(), "Mismatch between test labels and all labels"
#             assert np.allclose(train_freq, class_freqs, rtol=0.05), "Test set difference greater than 5%"
#             print(f"{foldname} ok. Class frequencies match")

#     def check_group_leakage(self, splits, metadata, split_on):
#         relevant_metadata = np.stack([metadata[key] for key in split_on], axis=1)
#         _, groups = np.unique(relevant_metadata, axis=0, return_inverse=True)
#         test_inds = splits.get("test", [])
#         test_groups = set(groups[test_inds])
#         print("\nChecking Group Leakage")
#         for foldname, folddict in splits.items():
#             if foldname == "test":
#                 continue
#             train_groups = set(groups[folddict["train"]])
#             val_groups = set(groups[folddict["val"]])
#             assert test_groups.isdisjoint(train_groups), "common groups between train and test"
#             assert test_groups.isdisjoint(val_groups), "common groups between val and test"
#             assert val_groups.isdisjoint(train_groups), "common groups between train and val"
#             print(f"{foldname} ok. No group leakage detected")
