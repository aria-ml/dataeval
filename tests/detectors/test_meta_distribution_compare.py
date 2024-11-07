import numpy as np
import pytest

from dataeval.detectors.ood.metadata_ks_compare import meta_distribution_compare


# Inputs with expected valid results:
@pytest.mark.parametrize(
    "md0, md1, expected",
    (
        (  # Basic valid inputs with one OOD example
            {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112], "random": [3.14, 159, 265]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111], "random": [1.12, 3.5, 8.13]},
            {
                "time": {"statistic_location": 0.44354838709677413, "shift_magnitude": 2.7, "pvalue": 0.0},
                "altitude": {
                    "statistic_location": 0.11612721970878584,
                    "shift_magnitude": 0.6598068274565396,
                    "pvalue": 0.9444444444444444,
                },
                "random": {
                    "statistic_location": 0.026565105350917086,
                    "shift_magnitude": 1.0549912166806692,
                    "pvalue": 0.22222222222222213,
                },
            },
        ),
        (  # Basic valid inputs with more than one OOD example
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {
                "time": {"statistic_location": 0.44354838709677413, "shift_magnitude": 2.7, "pvalue": 0.0},
                "altitude": {
                    "statistic_location": 0.11612721970878584,
                    "shift_magnitude": 0.6598068274565396,
                    "pvalue": 0.9444444444444444,
                },
            },
        ),
        # (  # Valid inputs: include non-numerical features.
        #     {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112], "weather": ["raining", "calm", "tornado"]},
        #     {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111], "weather": ["snow", "hail", "hot"]},
        #     np.array([True, True, False]),
        #     [("time", 2.0), ("time", 2.590909)],
        # ),
        # (  # Valid inputs: include random feature.
        #     {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112], "random": [3.14, 159, 265]},
        #     {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111], "random": [1.12, 3.5, 8.13]},
        #     np.array([True, True, False]),
        #     [("time", 2.0), ("time", 2.590909)],
        # ),
    ),
)
def test_output_values(md0, md1, expected: dict[str, dict[str, float]]):
    output = meta_distribution_compare(md0, md1)
    good = True
    for (ko, do), (ke, de) in zip(output.items(), expected.items()):
        good = good and all(
            (ke == k and np.isclose(ve, v, equal_nan=True)) for (ke, ve), (k, v) in zip(de.items(), do.items())
        )
    assert good


# # Inputs that raise Exceptions
# @pytest.mark.parametrize(
#     "md0, md1, error_msg",
#     (
#         (  # is_ood does not match metadata.
#             {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
#             {"time": np.array([42, 47]), "altitude": [235, 6789]},
#             "is_ood flag must have same length as new metadata 2 but has length 3.",
#         ),
#         (  # key mismatch.
#             {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
#             {"time": np.array([42, 47]), "schmaltitude": [235, 6789]},
#             re.escape(
#                 "Reference and test metadata keys must be identical: ['time', 'altitude'], ['time', 'schmaltitude']"
#             ),
#         ),
#         (  # wrong number of examples in a feature
#             {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]},
#             {"time": [7.8, 9.10], "altitude": [532, 9876, -2111]},
#             re.escape("All features must have same length, got lengths {3}, {2, 3}"),
#         ),
#     ),
# )
# def test_invalid_inputs(md0, md1, error_msg):
#     with pytest.raises(ValueError, match=error_msg):
#         meta_distribution_compare(md0, md1)


# # inputs that raise a warning
# @pytest.mark.parametrize(
#     "md0, md1, warning",
#     (
#         (  # Invalid inputs: md0 not enough examples.
#             {"time": 42, "altitude": 0},
#             {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
#             "We need at least 3 reference metadata examples to determine which features are least likely, but only got 1",  # noqa: E501
#         ),
#     ),
# )
# def test_nonsense_inputs(md0, md1, warning):
#     with pytest.warns(UserWarning, match=warning):
#         meta_distribution_compare(md0, md1)


# # Use a more realistic number of samples
# def test_bigdata_unlikely_features():
#     nbig = 1000
#     feature_names = ["temperature", "DJIA", "uptime"]
#     bigdata_size = (nbig, len(feature_names))
#     rng = np.random.default_rng(123)  # same pseudorandom sets each time.
#     X0 = rng.normal(size=bigdata_size)
#     X1 = rng.normal(size=bigdata_size)

#     half = int(nbig / 2)
#     X1[:half, 1] -= rng.normal(loc=5000, scale=200, size=half)  # first half will have weird DJIA
#     X1[half:, 0] += rng.normal(loc=100, scale=10, size=half)  # second half will have weird temperature

#     bigrefmetadata = {}
#     bignewmetadata = {}
#     for i, k in enumerate(feature_names):
#         bigrefmetadata.update({k: X0[:, i]})
#         bignewmetadata.update({k: X1[:, i]})

#     output = meta_distribution_compare(bigrefmetadata, bignewmetadata)

#     assert all(out[0] == "DJIA" for out in output[0:half]) and all(out[0] == "temperature" for out in output[half:])
