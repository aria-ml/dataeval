# eval with python -m pytest test_metadataparity.py
import warnings

import numpy as np
import pytest

import dataeval._internal.functional.metadataparity as metadataparity
from dataeval._internal.metrics.metadataparity import MetadataParity


class TestMDParityUnit:
    def test_warns_with_not_enough_frequency(self):
        factors = {
            "class": np.array(
                [
                    0,
                    1,
                ]
            ),
            "factor1": np.array([10, 20]),
        }

        with pytest.warns():
            mdp = MetadataParity(factors)
            mdp.evaluate()

    def test_passes_with_enough_frequency(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mdp = MetadataParity(factors)
            mdp.evaluate()

    def test_cant_quantize_strings(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate((["a"] * 5, ["b"] * 5)),
        }
        cfactors = np.array(["factor1"])
        bincounts = np.array([2])

        with pytest.raises(TypeError):
            mdp = MetadataParity(factors, continuous_factor_names=cfactors, continuous_factor_bincounts=bincounts)
            mdp.evaluate()

    def test_bad_factor_ref(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate((["a"] * 5, ["b"] * 5)),
        }
        cfactors = np.array(["something_else"])
        bincounts = np.array([2])

        with pytest.raises(Exception):
            MetadataParity(factors, continuous_factor_names=cfactors, continuous_factor_bincounts=bincounts)

    def test_unequal_cfactor_bincount_lengths(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate((["a"] * 5, ["b"] * 5)),
        }
        cfactors = np.array(["factor1"])
        bincounts = np.array([2, 3])

        with pytest.raises(ValueError):
            MetadataParity(factors, continuous_factor_names=cfactors, continuous_factor_bincounts=bincounts)

    def test_uneven_factor_lengths(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["a"] * 10),
            "factor2": np.array(["a"] * 11),
        }

        with pytest.raises(ValueError):
            MetadataParity(factors)

    def test_converts_output_correctly(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        mdp = MetadataParity(factors)
        chi_functional, p_functional = metadataparity.compute_parity(mdp.metadata_factors, mdp.labels)
        mdp_output = mdp.evaluate()
        chi_mdp, p_mdp = mdp_output["chi_squares"], mdp_output["p_values"]
        assert chi_functional == chi_mdp
        assert p_functional == p_mdp


class TestMDParityFunctional:
    def test_correlated_factors(self):
        """
        In this dataset, class and factor1 are perfectly correlated.
        This tests that the p-value is less than 0.05, which
        corresponds to class and factor1 being highly correlated.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 5, [20] * 5)),
        }

        mdp = MetadataParity(factors)
        output = mdp.evaluate()
        _, p = output["chi_squares"], output["p_values"]

        # Checks that factor1 is highly correlated with class
        assert p[0] < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogeneous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        mdp = MetadataParity(factors)
        output = mdp.evaluate()
        chi, p = output["chi_squares"], output["p_values"]

        # Checks that factor1 is uncorrelated with class
        assert np.isclose(chi[0], 0)
        assert np.isclose(p[0], 1)

    def test_quantized_factors(self):
        """
        This discretizes 'factor1' into having two values.
        This verifies that the '11' and '10' values get grouped together.
        """
        continuous_dataset = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 2, [11] * 3, [20] * 5)),
        }
        cfactors = np.array(["factor1"])
        bincounts = np.array([2])
        mdp = MetadataParity(
            continuous_dataset, continuous_factor_names=cfactors, continuous_factor_bincounts=bincounts
        )

        output1 = mdp.evaluate()
        chi1, p1 = output1["chi_squares"], output1["p_values"]

        discrete_dataset = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor2": np.concatenate(([10] * 5, [20] * 5)),
        }
        mdp = MetadataParity(discrete_dataset)

        output2 = mdp.evaluate()
        chi2, p2 = output2["chi_squares"], output2["p_values"]

        # Checks that the test on the quantization continuous_dataset is
        # equivalent to the test on the discrete dataset discrete_dataset
        assert p1[0] == p2[0]
        assert chi1[0] == chi2[0]

    def test_overquantized_factors(self):
        """
        This quantizes factor1 to have only one value, so that the discretized
        factor1 is the same over the entire dataset.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 5, [20] * 5)),
        }
        cfactors = np.array(["factor1"])
        bincounts = np.array([1])

        mdp = MetadataParity(factors, continuous_factor_names=cfactors, continuous_factor_bincounts=bincounts)
        output = mdp.evaluate()
        chi, p = output["chi_squares"], output["p_values"]

        # Checks if factor1 and class are perfectly uncorrelated
        assert np.isclose(chi[0], 0)
        assert np.isclose(p[0], 1)

    def test_underquantized_has_low_freqs(self):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 4, [15], [20] * 5)),
        }
        cfactors = np.array(["factor1"])
        bincounts = np.array([100])

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns():
            mdp = MetadataParity(factors, continuous_factor_names=cfactors, continuous_factor_bincounts=bincounts)
            mdp.evaluate()
