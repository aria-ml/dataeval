import numpy as np
import pytest

from src.dataeval.utils.split_dataset import split_dataset

####################################################################################################
############################################ Fixtures ##############################################
####################################################################################################

rng = np.random.default_rng(9251990)


@pytest.fixture(scope="module")
def labels():
    lbls = [np.full(50, 0), np.full(100, 1), np.full(125, 2), np.full(150, 3), np.full(75, 4)]
    return np.concatenate(lbls).astype(np.intp)


def make_groups(n_labels, num_folds, discrete, as_angle):
    n_groups = num_folds + 3
    as_angle = [as_angle] if isinstance(as_angle, bool) else as_angle
    discrete = [discrete] if isinstance(discrete, bool) else discrete
    group_dict = {}
    for angle in as_angle:
        for dis in discrete:
            group_enum = np.linspace(0, 360, n_groups).astype(np.intp)
            groups = rng.choice(group_enum, size=n_labels, replace=True)
            groups = groups if dis else rng.normal(loc=groups)
            group_key = ("Discrete" if dis else "Continuous") + ("_Angle" if angle else "_Int")
            group_dict[group_key] = groups
    return group_dict


####################################################################################################
###################################### Eval Helper Functions #######################################
####################################################################################################


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


def check_stratification(labels, splits, tolerance):
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
            atol=tolerance,  # , atol=1 / len(class_freqs)
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
            atol=tolerance,  # , atol=1 / len(class_freqs)
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


####################################################################################################
################################### Test Bad Input Arguments #######################################
####################################################################################################


def test_missing_val_frac_arg(labels):
    with pytest.raises(ValueError, match="If num_folds is None or 1, must assign a value to val_frac"):
        split_dataset(labels)
    with pytest.raises(ValueError, match="If num_folds is None or 1, must assign a value to val_frac"):
        split_dataset(labels, num_folds=1)


def test_invalid_val_frac_arg(labels):
    with pytest.raises(ValueError, match="If specifying val_frac, num_folds must be None or 1"):
        split_dataset(labels, num_folds=2, val_frac=0.25)


def test_missing_metadata(labels):
    with pytest.raises(UnboundLocalError, match="If split_on is specified, metadata must also be provided"):
        split_dataset(labels, num_folds=2, split_on=["Discrete_Int"], metadata=None)
    with pytest.raises(UnboundLocalError, match="If split_on is specified, metadata must also be provided"):
        split_dataset(labels, num_folds=2, split_on=["Discrete_Int"])


def test_stratification_warning(labels):
    with pytest.warns(UserWarning):
        split_dataset(labels, num_folds=labels.shape[0] // 2, stratify=True)


def test_continuous_labels(labels):
    cont_labels = labels.astype(np.float64) + np.random.uniform(-1, 1, size=labels.shape)
    error_statement = "Detected continuous labels, labels must be discrete for proper stratification"
    with pytest.raises(ValueError, match=error_statement):
        split_dataset(cont_labels, num_folds=2)


def test_too_many_partitions(labels):
    error_statement = f"""Total number of labels must greater than the number of total partitions.
            Got {len(labels)} labels and {labels.shape[0]+1} total train/val/test partitions."""
    with pytest.raises(IndexError, match=error_statement):
        split_dataset(labels, num_folds=labels.shape[0] + 1)


def test_label_metadata_mismatch(labels):
    metadata = make_groups(n_labels=labels.shape[0] + 1, num_folds=2, discrete=True, as_angle=False)
    error_statement = f"""Feature length does not match number of labels. 
                             Got {labels.shape[0]+1} features and {labels.shape[0]} samples"""
    with pytest.raises(IndexError, match=error_statement):
        split_dataset(labels, num_folds=2, split_on=["Discrete_Int"], metadata=metadata)


####################################################################################################
################################### Test Folding Done Correctly ####################################
####################################################################################################


@pytest.mark.parametrize("num_folds", [1, 5, 10])
@pytest.mark.parametrize("test_frac", [None, 0.25, 0.5])
def test_stratification(labels, num_folds, test_frac):
    val_frac = 0.1 if num_folds == 1 else None
    splits = split_dataset(labels, num_folds=num_folds, stratify=True, val_frac=val_frac, test_frac=test_frac)
    check_sample_leakage(splits)
    check_stratification(labels, splits, tolerance=0.01)


@pytest.mark.parametrize("num_folds", [1, 5, 10])
@pytest.mark.parametrize("group_angles", [False, [True, False]])
@pytest.mark.parametrize("group_discrete", [False, True])
@pytest.mark.parametrize("test_frac", [None, 0.25])
def test_grouping(labels, num_folds, group_angles, group_discrete, test_frac):
    val_frac = 0.1 if num_folds == 1 else None
    groups = make_groups(labels.shape[0], num_folds, group_discrete, group_angles)
    keys = list(groups.keys())
    splits = split_dataset(
        labels,
        num_folds=num_folds,
        stratify=False,
        split_on=keys,
        metadata=groups,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    check_sample_leakage(splits)
    check_group_leakage(splits, groups, keys)


@pytest.mark.parametrize("num_folds", [1, 5, 10])
@pytest.mark.parametrize("group_angles", [False, True])
@pytest.mark.parametrize("group_discrete", [False, True])
@pytest.mark.parametrize("test_frac", [None, 0.25])
def test_grouped_stratification(labels, num_folds, group_angles, group_discrete, test_frac):
    val_frac = 0.1 if num_folds == 1 else None
    groups = make_groups(labels.shape[0], num_folds, group_discrete, group_angles)
    keys = list(groups.keys())
    splits = split_dataset(
        labels,
        num_folds=num_folds,
        stratify=True,
        split_on=keys,
        metadata=groups,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    check_sample_leakage(splits)
    check_group_leakage(splits, groups, keys)
    check_stratification(labels, splits, tolerance=0.1)
