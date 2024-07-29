# eval with python -m pytest test_metadataparity.py
import warnings

import numpy as np
import pytest
from metadataparity import MetadataParity


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
            mdp = MetadataParity()
            mdp.set_factors(factors)
            chi, p = mdp.evaluate()

    def test_passes_with_enough_frequency(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mdp = MetadataParity()
            mdp.set_factors(factors)
            chi, p = mdp.evaluate()
    
    def test_cant_quantize_strings(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate((["a"] * 5, ["b"] * 5)),
        }
        cfactors = ["factor1"]
        bincounts = [2]

        with pytest.raises(TypeError):
            mdp = MetadataParity()
            mdp.set_factors(factors, continuous_factor_names = cfactors, continuous_factor_bincounts = bincounts)
            mdp.evaluate()
            

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

        mdp = MetadataParity()
        mdp.set_factors(factors)
        chi, p = mdp.evaluate()

        # Checks that factor1 is highly correlated with class
        assert p[0] < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogenous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        mdp = MetadataParity()
        mdp.set_factors(factors)
        chi, p = mdp.evaluate()

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
        cfactors = ["factor1"]
        bincounts = [2]
        mdp = MetadataParity()
        mdp.set_factors(continuous_dataset, continuous_factor_names = cfactors, continuous_factor_bincounts = bincounts)

        chi1, p1 = mdp.evaluate()

        discrete_dataset = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor2": np.concatenate(([10] * 5, [20] * 5)),
        }
        mdp = MetadataParity()
        mdp.set_factors(discrete_dataset)


        chi2, p2 = mdp.evaluate()

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
        cfactors = ["factor1"]
        bincounts = [1]

        mdp = MetadataParity()
        mdp.set_factors(factors, continuous_factor_names = cfactors, continuous_factor_bincounts = bincounts)
        chi, p = mdp.evaluate()

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
        cfactors = ["factor1"]
        bincounts = [100]

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns():
            mdp = MetadataParity()
            mdp.set_factors(factors, continuous_factor_names = cfactors, continuous_factor_bincounts = bincounts)
            chi, p = mdp.evaluate()