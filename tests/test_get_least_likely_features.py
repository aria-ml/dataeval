import numpy as np
import pytest

from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features

# @pytest.fixture(scope="class")
# def mock_llf():
#     md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
#     md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]}

#     is_ood = np.array([True, True, True])

#     return md0, md1, is_ood


# @pytest.fixture(scope="class")
# def mock_scalar():
#     md0 = {"time": 42, "altitude": 0}
#     md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]}

#     is_ood = np.array([True, True, True])

#     return md0, md1, is_ood


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
        (  # Basic valid inputs with one OOD example, scalar test case
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": 42, "altitude": 0},
            np.array([True]),
            [("time", 16.287128)],
        ),
    ),
)

# INPUT CHECKS:
# large enough md0
# numerical values in md0 and md1, not str etc
# scalar md1
# at least one ood
# md1 and is_ood same size


def test_output_values(md0, md1, is_ood, expected):
    output = get_least_likely_features(md0, md1, is_ood)
    print(f"################## {output}", flush=True)
    assert all((ke == k and np.isclose(ve, v)) for (ke, ve), (k, v) in zip(expected, output))
