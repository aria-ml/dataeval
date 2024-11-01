import numpy as np
import pytest

from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features


# Inputs with expected valid results:
@pytest.mark.parametrize(
    "md0, md1, is_ood, expected",
    (
        (  # Basic valid inputs with one OOD example
            {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([False, False, True]),
            [("time", 3.509091)],
        ),
        (  # Basic valid inputs with more than one OOD example
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([True, True, False]),
            [("time", 2.0), ("time", 2.590909)],
        ),
        (  # Basic valid inputs with more than one OOD example, with is_ood a list
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            [True, True, False],
            [("time", 2.0), ("time", 2.590909)],
        ),
        (  # Basic valid inputs with one OOD example, scalar test case
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": 42, "altitude": 0},
            np.array([True]),
            [("time", 16.287128)],
        ),
        (  # Valid inputs: no OOD examples.
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            np.array([False, False, False]),
            [],
        ),
    ),
)
def test_output_values(md0, md1, is_ood, expected: list[tuple[str, float]]):
    output = get_least_likely_features(md0, md1, is_ood)
    assert all((ke == k and np.isclose(ve, v, equal_nan=True)) for (ke, ve), (k, v) in zip(expected, output))


# Inputs that raise Exceptions
@pytest.mark.parametrize(
    "md0, md1, is_ood, expected",
    (
        (  # Invalid inputs: md0 not enough examples.
            {"time": 42, "altitude": 0},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([True, False, True]),
            ValueError,
        ),
        (  # is_ood does not match metadata.
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": np.array([42, 47]), "altitude": [235, 6789]},
            np.array([False, True, False]),
            ValueError,
        ),
    ),
)
def test_invalid_inputs(md0, md1, is_ood, expected: type):
    with pytest.raises(expected):
        _ = get_least_likely_features(md0, md1, is_ood)


# With a more realistic number of samples, make sure that
def test_bigdata_unlikely_features():
    nbig = 1000
    feature_names = ["temperature", "DJIA", "uptime"]
    bigdata_size = (nbig, len(feature_names))
    rng = np.random.default_rng(123)  # same pseudorandom sets each time.
    X0 = rng.normal(size=bigdata_size)
    X1 = rng.normal(size=bigdata_size)

    half = int(nbig / 2)
    X1[:half, 1] -= rng.normal(loc=5000, scale=200, size=half)  # first half will have weird DJIA
    X1[half:, 0] += rng.normal(loc=100, scale=10, size=half)  # second half will have weird temperature

    bigrefmetadata = {}
    bignewmetadata = {}
    for i, k in enumerate(feature_names):
        bigrefmetadata.update({k: X0[:, i]})
        bignewmetadata.update({k: X1[:, i]})

    is_ood = np.array([True] * nbig)
    output = get_least_likely_features(bigrefmetadata, bignewmetadata, is_ood)

    assert all(out[0] == "DJIA" for out in output[0:half]) and all(out[0] == "temperature" for out in output[half:])
