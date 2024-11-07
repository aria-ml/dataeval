import re

import numpy as np
import pytest

from dataeval.detectors.ood.metadata_ood_mi import get_metadata_ood_mi


# Inputs with expected valid results:
@pytest.mark.parametrize(
    "md0, is_ood, expected",
    (
        (  # Basic valid inputs
            {"time": np.linspace(0, 10, 100), "altitude": np.linspace(0, 16, 100) ** 2},
            np.array([True] * 62 + [False] * 38),
            {"time": 0.9359596758173668, "altitude": 0.9407686591507002},
        ),
        (  # Basic valid inputs, but one is non-numerical, which should just be skipped
            {"time": np.linspace(0, 10, 100), "name": ["phil"] * 100, "altitude": np.linspace(0, 16, 100) ** 2},
            np.array([True] * 62 + [False] * 38),
            {"time": 0.9359596758173668, "altitude": 0.9407686591507002},
        ),
        # (  # Basic valid inputs with different numbers of examples
        #     {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
        #     {"time": [7.8, 9.10, 11.12, 13.14], "altitude": [532, 9876, -2111, 4321]},
        #     {
        #         "time": {
        #             "statistic_location": 0.36850921273031817,
        #             "shift_magnitude": 3.131818181818182,
        #             "pvalue": 0.0,
        #         },
        #         "altitude": {
        #             "statistic_location": 0.06231169409918332,
        #             "shift_magnitude": 0.6530791624123108,
        #             "pvalue": 0.7777777777777777,
        #         },
        #     },
        # ),
        # (  # Valid inputs: include non-numerical features.
        #     {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112], "weather": ["raining", "calm", "tornado"]},
        #     {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111], "weather": ["snow", "hail", "hot"]},
        #     {
        #         "time": {"statistic_location": 0.44354838709677413, "shift_magnitude": 2.7, "pvalue": 0.0},
        #         "altitude": {
        #             "statistic_location": 0.11612721970878584,
        #             "shift_magnitude": 0.6598068274565396,
        #             "pvalue": 0.9444444444444444,
        #         },
        #         "weather": {},
        #     },
        # ),
        # (  # Valid inputs: feature with only one value
        #     {"time": [1.2, 1.2, 1.2], "altitude": [235, 6789, 101112], "random": [3.14, 159, 265]},
        #     {"time": [1.2, 1.2, 1.2], "altitude": [532, 9876, -2111], "random": [1.12, 3.5, 8.13]},
        #     {
        #         "time": {"statistic_location": 0.0, "shift_magnitude": 0.0, "pvalue": 1.0},
        #         "altitude": {
        #             "statistic_location": 0.11612721970878584,
        #             "shift_magnitude": 0.6598068274565396,
        #             "pvalue": 0.9444444444444444,
        #         },
        #         "random": {
        #             "statistic_location": 0.026565105350917086,
        #             "shift_magnitude": 1.0549912166806692,
        #             "pvalue": 0.22222222222222213,
        #         },
        #     },
        # ),
    ),
)
def test_output_values(md0, is_ood, expected: dict[str, float]):
    output = get_metadata_ood_mi(md0, is_ood)
    print("\n", flush=True)
    for k in output:
        print(f"{k}: {output[k]}, {expected[k]}", flush=True)

    assert all(np.isclose(output[k], expected[k]) for k in output)


# # Inputs that raise Exceptions
@pytest.mark.parametrize(
    "md0, is_ood, error_msg",
    (
        (  # key mismatch.
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            np.array([False, False, True, False]),
            re.escape("OOD flag and metadata features need to be same size, but are different sizes: 4 and 3."),
        ),
        # wrong number of examples in a feature
        (
            {"time": [7.8, 9.10], "altitude": [532, 9876, -2111]},
            np.array([True, False, False]),
            re.escape("Metadata features have differing sizes: {2, 3}"),
        ),
    ),
)
def test_invalid_inputs(md0, is_ood, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        get_metadata_ood_mi(md0, is_ood)


# # inputs that raise a warning
@pytest.mark.parametrize(
    "md0, is_ood, warning",
    (
        (  # Basic valid inputs
            {"time": np.linspace(0, 10, 100), "stuff": ["junk"] * 100, "altitude": np.linspace(0, 16, 100) ** 2},
            np.array([True] * 62 + [False] * 38),
            re.escape("Processing ['time', 'altitude'], others are non-numerical and will be skipped."),
        ),
    ),
)
def test_nonsense_inputs(md0, is_ood, warning):
    with pytest.warns(UserWarning, match=warning):
        get_metadata_ood_mi(md0, is_ood)
