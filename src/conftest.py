"""doctest fixtures"""

import pathlib
import sys
from typing import Mapping, Sequence

from dataeval.outputs._ood import OODOutput
from dataeval.utils.data._dataset import to_object_detection_dataset
from dataeval.utils.data._metadata import Metadata
from dataeval.utils.data._targets import Targets

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "detectors"))

from unittest.mock import MagicMock

import numpy as np
import pytest
import sklearn.datasets as dsets
import torch
from test_drift_uncertainty import PtModel

from dataeval.metrics.stats import hashstats, pixelstats
from dataeval.utils.torch.models import Autoencoder

# Set numpy print option to legacy 1.25 so native numpy types
# are not printed with dtype information.
if np.__version__[0] == "2":
    # WITHOUT LEGACY=1.25
    # >>> np.int32(16)
    # np.int32(16)

    # WITH LEGACY=1.25
    # >>> np.int32(16)
    # 16
    np.set_printoptions(legacy="1.25", precision=3)  # type: ignore


# Set manual seeds
np.random.seed(0)
torch.manual_seed(0)


def generate_random_metadata(
    labels: Sequence[str], factors: Mapping[str, Sequence[str]], length: int, random_seed: int
) -> Metadata:
    rng = np.random.default_rng(random_seed)
    labels_arr = rng.choice(range(len(labels)), (length))
    scores_arr = np.eye(len(labels))[labels_arr].astype(np.float32)
    targets = Targets(labels_arr, scores_arr, None, None)
    metadata_dict = {k: list(rng.choice(v, (length))) for k, v in factors.items()}
    metadata = Metadata(None)  # type: ignore
    metadata._raw = [{} for _ in range(len(labels))]
    metadata._collated = True
    metadata._targets = targets
    metadata._class_labels = targets.labels
    metadata._class_names = list(labels)
    metadata._merged = metadata_dict, {}
    return metadata


class ClassificationModel(PtModel):
    def __init__(self):
        super().__init__(16, 3, True)


@pytest.fixture(autouse=True, scope="session")
def add_all(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["generate_random_metadata"] = generate_random_metadata
    doctest_namespace["OODOutput"] = OODOutput


@pytest.fixture(autouse=True, scope="session")
def doctest_metadata_explanatory_funcs(doctest_namespace):
    md1 = MagicMock(spec=Metadata)
    md2 = MagicMock(spec=Metadata)

    md1.discrete_factor_names = []
    md1.continuous_factor_names = ["time", "altitude"]
    md1.continuous_data = np.array([[1.2, 235], [3.4, 6789], [5.6, 101112]])
    md1.total_num_factors = 2

    md2.discrete_factor_names = []
    md2.continuous_factor_names = ["time", "altitude"]
    md2.continuous_data = np.array([[7.8, 532], [9.10, 9876], [11.12, 211101]])
    md2.total_num_factors = 2

    doctest_namespace["metadata1"] = md1
    doctest_namespace["metadata2"] = md2


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_linters_duplicates(doctest_namespace):
    rng = np.random.default_rng(273)
    base = np.concatenate([np.ones((5, 10)), np.zeros((5, 10))])
    images = np.stack([rng.permutation(base) * i for i in range(50)], axis=0)
    images[16] = images[37]
    images[3] = images[20]

    """dataeval.detectors.linters.Duplicates"""

    doctest_namespace["duplicate_images"] = images
    doctest_namespace["hashes1"] = hashstats(images[:24])
    doctest_namespace["hashes2"] = hashstats(images[25:])


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_linters_outliers(doctest_namespace):
    images = np.ones((30, 1, 128, 128), dtype=np.int32) * 2
    images = images + np.repeat(np.arange(10), 3 * 128 * 128).reshape(30, -1, 128, 128)
    images[10:13, :, 50:80, 50:80] = 0
    images[[7, 11, 18, 25]] = 512

    """dataeval.detectors.linters.Outliers"""

    doctest_namespace["outlier_images"] = images
    doctest_namespace["stats1"] = pixelstats(images[:14])
    doctest_namespace["stats2"] = pixelstats(images[15:])


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_ood_drift(doctest_namespace):
    train_images = np.zeros((50, 1, 16, 16), dtype=np.float32)
    test_images = np.ones((8, 1, 16, 16), dtype=np.float32)
    test_images[2] = 0

    """dataeval.detectors.ood.OOD_AE"""
    """dataeval.detectors.drift.DriftKS"""
    """dataeval.detectors.drift.DriftCVM"""
    """dataeval.detectors.drift.DriftMMD"""

    doctest_namespace["train_images"] = train_images
    doctest_namespace["test_images"] = test_images
    doctest_namespace["encoder"] = Autoencoder(1)


@pytest.fixture(autouse=True, scope="session")
def doctest_detectors_drift_uncertainty(doctest_namespace):
    x_ref = np.random.randn(*(500, 16)).astype(np.float32)
    x_test = np.ones_like(x_ref)
    doctest_namespace["ClassificationModel"] = ClassificationModel
    doctest_namespace["x_ref"] = x_ref
    doctest_namespace["x_test"] = x_test


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_bias_balance_diversity(doctest_namespace):
    str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
    class_labels = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    cnt_vals = [-0.54, -0.32, 0.41, 1.04, -0.13, 1.37, -0.67, 0.35, 0.90, 0.09, -0.74, -0.92]
    cat_vals = [1.1, 1.1, 0, 0, 1.1, 0, 1.1, 0, 0, 1.1, 1.1, 0]
    metadata_dict = {"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals}
    continuous_factor_bincounts = {"var_cnt": 5, "var_float_cat": 2}
    labels = np.asarray(class_labels)
    scores = np.eye(2)[labels].astype(np.float32)
    targets = Targets(labels=labels, scores=scores, bboxes=None, source=None)
    metadata = Metadata(None, continuous_factor_bins=continuous_factor_bincounts)  # type: ignore
    metadata._collated = True
    metadata._raw = [{} for _ in range(len(class_labels))]
    metadata._targets = targets
    metadata._class_labels = np.asarray(class_labels)
    metadata._class_names = ["cat", "dog"]
    metadata._merged = metadata_dict, {}

    """dataeval.metrics.bias.balance.balance"""
    """dataeval.metrics.bias.diversity.diversity"""

    doctest_namespace["metadata"] = metadata


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_bias_coverage(doctest_namespace):
    blobs = dsets.make_blobs(n_samples=500, centers=np.array([(1, 1), (3, 3)]), cluster_std=0.5, random_state=498)
    blobs = np.asarray(blobs[0], dtype=np.float64)
    blobs = blobs - np.min(blobs)
    blobs = blobs / np.max(blobs)

    """dataeval.metrics.bias.coverage.coverage"""
    doctest_namespace["embeddings"] = blobs


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_estimators_clusterer(doctest_namespace):
    images = dsets.make_blobs(n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.5, random_state=33)[0]
    images[9] = images[24]
    images[23] = images[48] + 1e-5

    """dataeval.metrics.estimators.clusterer"""
    doctest_namespace["clusterer_images"] = images


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_estimators_divergence(doctest_namespace):
    a = dsets.make_blobs(n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.3, random_state=712)[0]
    b = dsets.make_blobs(n_samples=50, centers=np.array([(-0.5, -0.5), (1, 1)]), cluster_std=0.3, random_state=712)[0]

    """dataeval.metrics.estimators.divergence.divergence"""

    doctest_namespace["datasetA"] = a
    doctest_namespace["datasetB"] = b


@pytest.fixture(autouse=True, scope="session")
def doctest_metrics_stats(doctest_namespace):
    images = np.repeat(np.arange(65536, dtype=np.int32), 4 * 8).reshape(8, -1, 128, 128)[:, :3, :, :]
    for i in range(8):
        for j in range(3):
            images[i, j, 30:50, 50:80] = i * j
    images = list(images)
    images[2] = np.resize(np.mean(images[2], axis=0), (1, 96, 128))
    images[4] = np.resize(np.mean(images[4], axis=0), (1, 96, 64))

    bboxes = [
        [[5, 21, 24, 43], [7, 4, 17, 21]],
        [[12, 23, 28, 24]],
        [[13, 9, 29, 23], [17, 7, 39, 20], [2, 14, 9, 26]],
        [[18, 14, 28, 29]],
        [[21, 18, 44, 27], [15, 13, 28, 23]],
        [[13, 2, 23, 14]],
        [[4, 16, 8, 20], [16, 14, 25, 29]],
        [[1, 22, 13, 45], [12, 20, 27, 21], [16, 22, 39, 28]],
    ]

    rng = np.random.default_rng(4)
    label_array = rng.choice(5, 50)
    labels = []
    for i, boxes in enumerate(bboxes):
        num_labels = len(boxes)
        selected_labels = label_array[5 * i : 5 * i + num_labels].tolist()
        labels.append(selected_labels)

    classes = ["horse", "cow", "sheep", "pig", "chicken"]

    """dataeval.metrics.stats.boxratiostats.boxratiostats"""
    """dataeval.metrics.stats.imagestats.imagestats"""
    """dataeval.metrics.stats.dimensionstats.dimensionstats"""
    """dataeval.metrics.stats.hashstats.hashstats"""
    """dataeval.metrics.stats.labelstats.labelstats"""
    """dataeval.metrics.stats.pixelstats.pixelstats"""
    """dataeval.metrics.stats.visualstats.visualstats"""

    doctest_namespace["dataset"] = to_object_detection_dataset(images, labels, bboxes, None, classes)


@pytest.fixture(autouse=True, scope="session")
def doctest_workflows_sufficiency(doctest_namespace):
    model = MagicMock()
    train_ds = MagicMock()
    train_ds.__len__.return_value = 100
    test_ds = MagicMock()
    test_ds.__len__.return_value = 10
    train_fn = MagicMock()
    eval_fn = MagicMock()
    eval_fn.return_value = {"test": 1.0}

    """dataeval.workflows.sufficiency.Sufficiency"""

    doctest_namespace["model"] = model
    doctest_namespace["train_ds"] = train_ds
    doctest_namespace["test_ds"] = test_ds
    doctest_namespace["train_fn"] = train_fn
    doctest_namespace["eval_fn"] = eval_fn


@pytest.fixture(autouse=True, scope="session")
def doctest_sampledataset(doctest_namespace):
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, size: int, class_count: int) -> None:
            self._size = size
            self._class_count = class_count

        def __getitem__(self, index: int) -> tuple[str, np.ndarray, dict[str, int]]:
            one_hot_label = np.zeros(10)
            one_hot_label[index % self._class_count] = 1
            return f"data_{index}", one_hot_label, {"id": index}

        def __len__(self) -> int:
            return self._size

    doctest_namespace["SampleDataset"] = SampleDataset
