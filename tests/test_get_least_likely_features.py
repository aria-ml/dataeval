import numpy as np
import pytest

from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features


@pytest.fixture(scope="class")
def mock_llf():
    md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]}

    is_ood = np.array([True, True, True])

    return md0, md1, is_ood


@pytest.fixture(scope="class")
def mock_scalar():
    md0 = {"time": 42, "altitude": 0}
    md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]}

    is_ood = np.array([True, True, True])

    return md0, md1, is_ood


@pytest.mark.parametrize(
    "md0, md1, is_ood, expected",
    (
        (
            {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([False, False, True]),
            [("time", 3.509091)],
        ),
        (
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": 42, "altitude": 0},
            np.array([False, False, True]),
            [("altitude", 2111.0)],
        ),
        (
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([False, False, True]),
            [("time", 3.509091)],
        ),
    ),
)
def test_output_values(md0, md1, is_ood, expected):
    output = get_least_likely_features(md0, md1, is_ood)
    print(f"################## {output}", flush=True)
    assert all((ke == k and np.isclose(ve, v)) for (ke, ve), (k, v) in zip(expected, output))
