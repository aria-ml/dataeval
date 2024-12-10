import re
import warnings

import numpy as np
import pytest

from dataeval.detectors.ood.metadata_ks_compare import meta_distribution_compare


# Inputs with expected valid results:
@pytest.mark.parametrize(
    "md0, md1, expected",
    (
        (  # Basic valid inputs
            {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112], "random": [3.14, 159, 265]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111], "random": [1.12, 3.5, 8.13]},
            {
                "time": {
                    "statistic": 1.0,
                    "statistic_location": 0.44354838709677413,
                    "shift_magnitude": 2.7,
                    "pvalue": 0.0,
                },
                "altitude": {
                    "statistic": 0.33333333333333337,
                    "statistic_location": 0.11612721970878584,
                    "shift_magnitude": 0.6598068274565396,
                    "pvalue": 0.9444444444444444,
                },
                "random": {
                    "statistic": 0.6666666666666667,
                    "statistic_location": 0.026565105350917086,
                    "shift_magnitude": 1.0549912166806692,
                    "pvalue": 0.22222222222222213,
                },
            },
        ),
        (  # Basic valid inputs with different numbers of examples
            {"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112]},
            {"time": [7.8, 9.10, 11.12, 13.14], "altitude": [532, 9876, -2111, 4321]},
            {
                "time": {
                    "statistic": 1.0,
                    "statistic_location": 0.36850921273031817,
                    "shift_magnitude": 3.131818181818182,
                    "pvalue": 0.0,
                },
                "altitude": {
                    "statistic": 0.4166666666666667,
                    "statistic_location": 0.06231169409918332,
                    "shift_magnitude": 0.6530791624123108,
                    "pvalue": 0.7777777777777777,
                },
            },
        ),
        (  # Valid inputs: include non-numerical features.
            {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112], "weather": ["raining", "calm", "tornado"]},
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111], "weather": ["snow", "hail", "hot"]},
            {
                "time": {
                    "statistic": 1.0,
                    "statistic_location": 0.44354838709677413,
                    "shift_magnitude": 2.7,
                    "pvalue": 0.0,
                },
                "altitude": {
                    "statistic": 0.33333333333333337,
                    "statistic_location": 0.11612721970878584,
                    "shift_magnitude": 0.6598068274565396,
                    "pvalue": 0.9444444444444444,
                },
                "weather": {},
            },
        ),
        (  # Valid inputs: feature with only one value
            {"time": [1.2, 1.2, 1.2], "altitude": [235, 6789, 101112], "random": [3.14, 159, 265]},
            {"time": [1.2, 1.2, 1.2], "altitude": [532, 9876, -2111], "random": [1.12, 3.5, 8.13]},
            {
                "time": {"statistic": 0.0, "statistic_location": 0.0, "shift_magnitude": 0.0, "pvalue": 1.0},
                "altitude": {
                    "statistic": 0.33333333333333337,
                    "statistic_location": 0.11612721970878584,
                    "shift_magnitude": 0.6598068274565396,
                    "pvalue": 0.9444444444444444,
                },
                "random": {
                    "statistic": 0.6666666666666667,
                    "statistic_location": 0.026565105350917086,
                    "shift_magnitude": 1.0549912166806692,
                    "pvalue": 0.22222222222222213,
                },
            },
        ),
    ),
)
def test_output_values(md0, md1, expected: dict[str, dict[str, float]]):
    output = meta_distribution_compare(md0, md1).mdc  # dict[str, MetadataKSResult]
    good = True
    for (kfo, ksout), (kfe, de) in zip(output.items(), expected.items()):
        good = (
            good
            and (kfo == kfe)
            and all(np.isclose(ve, ksq, equal_nan=True) for (ke, ve), ksq in zip(de.items(), ksout))
        )
    assert good


# # Inputs that raise Exceptions
@pytest.mark.parametrize(
    "md0, md1, error_msg",
    (
        (  # key mismatch.
            {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]},
            {"time": np.array([42, 47]), "schmaltitude": [235, 6789]},
            re.escape("Both sets of metadata keys must be identical: ['time', 'altitude'], ['time', 'schmaltitude']"),
        ),
        # (  # wrong number of examples in a feature
        #     {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]},
        #     {"time": [7.8, 9.10], "altitude": [532, 9876, -2111]},
        #     re.escape("All features must have same length, got lengths {3}, {2, 3}"),
        # ),
    ),
)
def test_invalid_inputs(md0, md1, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        meta_distribution_compare(md0, md1)


# # inputs that raise a warning
@pytest.mark.parametrize(
    "md0, md1, warning",
    (
        (  # Invalid inputs: md0 not enough examples.
            {"time": [42], "altitude": [532, 9876, -2111, 42, 83, 314159, 16, 99]},
            {
                "time": [7.8, 9.10, 11.12, 13.14, 15.16, 17.18, 19.20, 21.22],
                "altitude": [532, 9876, -2111, 42, 83, 314159, 16, 99],
            },
            "Sample sizes of 1, 8 for feature time will yield unreliable p-values from the KS test.",
        ),
    ),
)
def test_nonsense_inputs(md0, md1, warning):
    with pytest.warns(UserWarning, match=warning):
        meta_distribution_compare(md0, md1)


# Use a more realistic number of samples; make sure no warning is emitted.
def test_bigdata_unlikely_features():
    with warnings.catch_warnings() as record:
        warnings.simplefilter("error")
        nbig = 1000
        feature_names = ["temperature", "DJIA", "uptime"]
        bigdata_size = (nbig, len(feature_names))
        rng = np.random.default_rng(4567)  # same pseudorandom sets each time.
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

        output = meta_distribution_compare(bigrefmetadata, bignewmetadata).mdc

        expected = {
            "temperature": {
                "statistic": 0.5,
                "statistic_location": 0.04942976244618348,
                "shift_magnitude": 37.37728630019768,
                "pvalue": 4.1132799581816557e-116,
            },
            "DJIA": {
                "statistic": 0.501,
                "statistic_location": 0.9988396927777546,
                "shift_magnitude": 1834.6649059091003,
                "pvalue": 1.3128633809722045e-116,
            },
            "uptime": {
                "statistic": 0.02799999999999997,
                "statistic_location": 0.4362350597356996,
                "shift_magnitude": 0.04229056583710559,
                "pvalue": 0.8172815627702071,
            },
        }
        good = True
        for (kfo, ksout), (kfe, de) in zip(output.items(), expected.items()):
            good = (
                good
                and (kfo == kfe)
                and all(np.isclose(ve, ksq, equal_nan=True) for (ke, ve), ksq in zip(de.items(), ksout))
            )
        assert good and record is None
