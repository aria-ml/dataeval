import pytest

from dataeval.utils.data import Metadata
from src.dataeval.utils.metadata import merge
from tests.conftest import preprocess

BIG_SAMPLES_COUNT = 1000
BIG_FEATURE_NAMES = ["DJIA", "temperature", "uptime"]


@pytest.fixture(scope="module")
def metadata_ref_big(RNG) -> Metadata:
    """
    Higher count Metadata values with a normal distribution
    """
    shape = (BIG_SAMPLES_COUNT, len(BIG_FEATURE_NAMES))

    # Generate data samples
    X = RNG.normal(size=shape)

    # Each key has 1000 samples of feature_i
    metadata = {k: X[:, i].tolist() for i, k in enumerate(BIG_FEATURE_NAMES)}

    MD = preprocess(
        merge([metadata]),
        class_labels=range(1000),
        continuous_factor_bins={k: len(v) for k, v in metadata.items()},
    )

    return MD


@pytest.fixture(scope="module")
def metadata_tst_big(RNG) -> Metadata:
    """
    Higher count Metadata values with two added random normal
    distributions in the first half and second half respectively
    """
    bigdata_size = (BIG_SAMPLES_COUNT, len(BIG_FEATURE_NAMES))

    # Generate data samples
    X = RNG.normal(size=bigdata_size)

    # Add noise
    half = BIG_SAMPLES_COUNT // 2
    X[:half, 0] -= RNG.normal(loc=5000, scale=200, size=half)
    X[half:, 1] += RNG.normal(loc=100, scale=10, size=half)

    # Each key has BIG_SAMPLES_COUNT of feature_i
    metadata_tst = {k: X[:, i].tolist() for i, k in enumerate(BIG_FEATURE_NAMES)}

    MD = preprocess(
        merge([metadata_tst]),
        class_labels=range(1000),
        continuous_factor_bins={k: len(v) for k, v in metadata_tst.items()},
    )

    return MD
