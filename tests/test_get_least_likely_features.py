import numpy as np
import pytest

from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features


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
        (  # Invalid inputs: md0 not enough examples.
            {"time": 42, "altitude": 0},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([True, False, True]),
            [("not enough reference metadata", np.nan)],
        ),
        (  # Valid inputs: no OOD examples.
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            np.array([False, False, False]),
            [("all examples are in-distribution", np.nan)],
        ),
        (  # is_ood does not match metadata.
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": np.array([42, 47]), "altitude": [235, 6789]},
            np.array([False, True, False]),
            [("is_ood flag must have same length as metadata.", np.nan)],
        ),
    ),
)
# INPUT CHECKS:
# X large enough md0
# X numerical values in md0 and md1, not str etc
# X scalar md1
# X at least one ood
# md1 and is_ood same size


def test_output_values(md0, md1, is_ood, expected):
    output = get_least_likely_features(md0, md1, is_ood)
    print(f"################## {output}", flush=True)
    assert all((ke == k and np.isclose(ve, v, equal_nan=True)) for (ke, ve), (k, v) in zip(expected, output))


def test_bigdata_unlikely_features():
    nbig = 1000
    feature_names = ["temperature", "DJIA", "uptime"]
    bigdata_size = (nbig, len(feature_names))
    rng = np.random.default_rng(123)
    X0 = rng.normal(size=bigdata_size)
    X1 = rng.normal(size=bigdata_size)

    half = int(nbig / 2)
    X1[:half, 1] -= 5000  # first half will have weird DJIA
    X1[half:, 0] += 30  # second half will have weird temperature

    bigrefmetadata = {}
    bignewmetadata = {}
    for i, k in enumerate(feature_names):
        bigrefmetadata.update({k: X0[:, i]})
        bignewmetadata.update({k: X1[:, i]})

    is_ood = np.array([True] * nbig)
    output = get_least_likely_features(bigrefmetadata, bignewmetadata, is_ood)

    print("@@@@@@@@@@@@@@@@@@@", output[0], output[-1])

    assert all(out[0] == "DJIA" for out in output[0:half]) and all(out[0] == "temperature" for out in output[half:])

    # (  # big data find weird feature test...
    #     bigrefmetadata,
    #     bignewmetadata,
    #     np.array([True]*nbig),
    #     [("is_ood flag must have same length as metadata.", np.nan)],
    # ),
