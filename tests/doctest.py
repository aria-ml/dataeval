import numpy as np
import pytest

# Set numpy print option to legacy 1.25 so native numpy types
# are not printed with dtype information.
if np.__version__[0] == "2":
    # WITHOUT LEGACY=1.25
    # >>> np.int32(16)
    # np.int32(16)

    # WITH LEGACY=1.25
    # >>> np.int32(16)
    # 16
    np.set_printoptions(legacy="1.25")  # type: ignore


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def doctest_detectors_linters_clusterer(doctest_namespace):
    import sklearn.datasets as dsets

    images = dsets.make_blobs(n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.5, random_state=33)[0]
    images[9] = images[24]
    images[23] = images[48] + 1e-5

    """dataeval.detectors.linters.Clusterer"""
    doctest_namespace["clusterer_images"] = images


@pytest.fixture(autouse=True)
def doctest_detectors_linters_duplicates(doctest_namespace):
    from dataeval.metrics.stats import hashstats

    rng = np.random.default_rng(273)
    base = np.concatenate([np.ones((5, 10)), np.zeros((5, 10))])
    images = np.stack([rng.permutation(base) * i for i in range(50)], axis=0)
    images[16] = images[37]
    images[3] = images[20]

    """dataeval.detectors.linters.Duplicates"""

    doctest_namespace["duplicate_images"] = images
    doctest_namespace["hashes1"] = hashstats(images[:24])
    doctest_namespace["hashes2"] = hashstats(images[25:])


@pytest.fixture(autouse=True)
def doctest_detectors_linters_outliers(doctest_namespace):
    from dataeval.metrics.stats import pixelstats

    images = np.ones((30, 1, 128, 128), dtype=np.int32) * 2
    images = images + np.repeat(np.arange(10), 3 * 128 * 128).reshape(30, -1, 128, 128)
    images[10:13, :, 50:80, 50:80] = 0
    images[[7, 11, 18, 25]] = 512

    """dataeval.detectors.linters.Outliers"""

    doctest_namespace["outlier_images"] = images
    doctest_namespace["stats1"] = pixelstats(images[:14])
    doctest_namespace["stats2"] = pixelstats(images[15:])


@pytest.fixture(autouse=True)
def doctest_metrics_bias_balance_diversity(doctest_namespace):
    from dataeval.utils.metadata import preprocess

    str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
    class_labels = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    cnt_vals = [-0.54, -0.32, 0.41, 1.04, -0.13, 1.37, -0.67, 0.35, 0.90, 0.09, -0.74, -0.92]
    cat_vals = [1.1, 1.1, 0, 0, 1.1, 0, 1.1, 0, 0, 1.1, 1.1, 0]
    metadata_dict = [{"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals}]
    continuous_factor_bincounts = {"var_cnt": 5, "var_float_cat": 2}
    metadata = preprocess(metadata_dict, class_labels, continuous_factor_bincounts)

    """dataeval.metrics.bias.balance.balance"""
    """dataeval.metrics.bias.diversity.diversity"""

    doctest_namespace["metadata"] = metadata


@pytest.fixture(autouse=True)
def doctest_metrics_bias_coverage(doctest_namespace):
    import sklearn.datasets as dsets

    blobs = dsets.make_blobs(n_samples=500, centers=np.array([(1, 1), (3, 3)]), cluster_std=0.5, random_state=498)

    """dataeval.metrics.bias.coverage.coverage"""

    doctest_namespace["embeddings"] = blobs[0]


@pytest.fixture(autouse=True)
def doctest_metrics_estimators_divergence(doctest_namespace):
    import sklearn.datasets as dsets

    a = dsets.make_blobs(n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.3, random_state=712)[0]
    b = dsets.make_blobs(n_samples=50, centers=np.array([(-0.5, -0.5), (1, 1)]), cluster_std=0.3, random_state=712)[0]

    """dataeval.metrics.estimators.divergence.divergence"""

    doctest_namespace["datasetA"] = a
    doctest_namespace["datasetB"] = b


@pytest.fixture(autouse=True)
def doctest_metrics_stats(doctest_namespace):
    from dataeval.metrics.stats import dimensionstats

    images = np.repeat(np.arange(65536, dtype=np.int32), 4 * 5).reshape(5, -1, 128, 128)[:, :3, :, :]
    for i in range(5):
        for j in range(3):
            images[i, j, 30:50, 50:80] = i * j
    images = list(images)
    images[2] = np.resize(np.mean(images[2], axis=0), (1, 96, 128))
    images[4] = np.resize(np.mean(images[4], axis=0), (1, 96, 64))

    bboxes = [
        np.array([[5, 21, 24, 43], [7, 4, 17, 21]]),
        np.array([[12, 23, 28, 24]]),
        np.array([[13, 9, 29, 23], [17, 7, 39, 20], [2, 14, 9, 26]]),
        np.array([[18, 14, 28, 29]]),
        np.array([[21, 18, 44, 27], [15, 13, 28, 23]]),
        np.array([[13, 2, 23, 14]]),
        np.array([[4, 16, 8, 20], [16, 14, 25, 29]]),
        np.array([[1, 22, 13, 45], [12, 20, 27, 21], [16, 22, 39, 28]]),
        np.array([[16, 5, 30, 13]]),
        np.array([[2, 18, 11, 30], [9, 22, 23, 42]]),
    ]

    rng = np.random.default_rng(4)
    label_array = rng.choice(["horse", "cow", "sheep", "pig", "chicken"], 50)
    labels = []
    for i in range(10):
        num_labels = rng.choice(5) + 1
        selected_labels = list(label_array[5 * i : 5 * i + num_labels])
        labels.append(selected_labels)

    """dataeval.metrics.stats.boxratiostats.boxratiostats"""
    """dataeval.metrics.stats.datasetstats.channelstats"""
    """dataeval.metrics.stats.datasetstats.datasetstats"""
    """dataeval.metrics.stats.dimensionstats.dimensionstats"""
    """dataeval.metrics.stats.hashstats.hashstats"""
    """dataeval.metrics.stats.labelstats.labelstats"""
    """dataeval.metrics.stats.pixelstats.pixelstats"""
    """dataeval.metrics.stats.visualstats.visualstats"""

    doctest_namespace["dimensionstats"] = dimensionstats
    doctest_namespace["stats_images"] = images
    doctest_namespace["bboxes"] = bboxes
    doctest_namespace["labels"] = labels


@pytest.fixture(autouse=True)
def doctest_workflows_sufficiency(doctest_namespace):
    from unittest.mock import MagicMock, patch

    import numpy as np

    model = MagicMock()
    train_ds = MagicMock()
    train_ds.__len__.return_value = 100
    test_ds = MagicMock()
    test_ds.__len__.return_value = 10
    train_fn = MagicMock()
    eval_fn = MagicMock()
    eval_fn.return_value = {"test": 1.0}

    mock_params = patch("dataeval.workflows.sufficiency.calc_params").start()
    mock_params.return_value = np.array([0.0, 42.0, 0.0])

    """dataeval.workflows.sufficiency.Sufficiency"""

    doctest_namespace["model"] = model
    doctest_namespace["train_ds"] = train_ds
    doctest_namespace["test_ds"] = test_ds
    doctest_namespace["train_fn"] = train_fn
    doctest_namespace["eval_fn"] = eval_fn