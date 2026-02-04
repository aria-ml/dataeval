import logging

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataeval.performance._output import Constraints, calc_params, f_out, linear_initialization

np.random.seed(0)
torch.manual_seed(0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(6400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def linear_fit_manual(x, y):
    """
    Basic helper for linear regression
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x**2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    b = (sum_y - m * sum_x) / n
    return m, b


@pytest.mark.required
class TestSufficiencyUtils:
    def test_f_out(self):
        n_i = np.geomspace(30, 3000, 20).astype(np.intp)
        x = np.array([0.5, 0.5, 0.5])
        answer = [
            0.591287,
            0.581111,
            0.572169,
            0.5635,
            0.556254,
            0.55,
            0.544194,
            0.539163,
            0.534669,
            0.530715,
            0.527196,
            0.524084,
            0.521339,
            0.518898,
            0.516741,
            0.514828,
            0.513135,
            0.511634,
            0.510305,
            0.509129,
        ]

        output = f_out(n_i, x)

        npt.assert_almost_equal(output, answer, decimal=5)

    def test_get_params(self):
        p_i = np.array(
            [
                0.806667,
                0.63,
                0.669333,
                0.673333,
                0.622,
                0.485333,
                0.399333,
                0.382667,
                0.330667,
                0.314,
                0.284667,
                0.256,
                0.228667,
                0.219333,
                0.202667,
                0.180667,
                0.168667,
                0.149333,
                0.141333,
                0.137333,
            ]
        )
        n_i = np.geomspace(30, 3000, 20).astype(np.intp)
        answer = [3.2633, 0.4160, 0.0053]
        output = calc_params(p_i, n_i, 100, True)

        npt.assert_almost_equal(output, answer, decimal=3)

    def test_linear_initialization_increasing(self):
        """
        Tests linear initialization for increasing power law curve
        """
        metric = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
        steps = np.array([20, 200, 2000, 2500, 3000])
        bounds = Constraints(scale=(None, None), negative_exponent=(0, None), asymptote=(None, None))
        x0 = linear_initialization(metric, steps, bounds)
        assert x0[2] == -0.001  # did not apply bounds
        m_log = np.log(metric - (-0.001))
        s_log = np.log(steps)
        m, b = linear_fit_manual(s_log, m_log)
        test = np.array([np.exp(b), -m, -0.001])
        assert np.allclose(
            x0,
            test,
            rtol=0,
            atol=1e-8,
        )
        # test values that produce a slope of 1, asymptote of 0, and coefficient of np.exp(-1.90776)
        metric = np.exp([-2.90776, -3.90776, -4.90776, -5.90776, -6.90776])
        steps = np.exp([1, 2, 3, 4, 5])
        x0 = linear_initialization(metric, steps, bounds)
        npt.assert_almost_equal(x0[0], np.exp(-1.90776), decimal=5)
        npt.assert_almost_equal(x0[1], 1, decimal=5)
        npt.assert_almost_equal(x0[2], 0, decimal=5)

    def test_linear_initialization_increasing_bounded(self):
        """
        Tests linear initialization for increasing power law curve with c0 bounded [0,1]
        """
        metric = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
        steps = np.array([20, 200, 2000, 2500, 3000])
        bounds = Constraints(scale=(None, None), negative_exponent=(0, None), asymptote=(0, 1))
        x0 = linear_initialization(metric, steps, bounds)
        assert x0[2] == 0  # applied bounds
        # apply y offset
        m_log = np.log(metric - (-0.001))
        s_log = np.log(steps)
        m, b = linear_fit_manual(s_log, m_log)
        test = np.array([np.exp(b), -m, 0])
        assert np.allclose(
            x0,
            test,
            rtol=0,
            atol=1e-8,
        )

    def test_linear_initialization_decreasing(self):
        """
        Tests linear initialization for decreasing power law curve
        """
        metric = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        steps = np.array([20, 200, 2000, 2500, 3000])
        bounds = Constraints(scale=(None, None), negative_exponent=(0, None), asymptote=(None, None))
        x0 = linear_initialization(metric, steps, bounds)
        assert x0[2] == 1.001  # did not apply bounds
        # apply y offset
        m_log = np.log((1.001) - metric)
        s_log = np.log(steps)
        # apply manual fit to linearized points for comparison
        m, b = linear_fit_manual(s_log, m_log)
        test = np.array([-np.exp(b), -m, 1.001])
        assert np.allclose(
            x0,
            test,
            rtol=0,
            atol=1e-8,
        )
        # test values that produce a slope of ~0, asymptote of 1, and coefficient of -(c_hat - exp(-0.00100050038))
        metric = np.exp([-0.00100050037, -0.00100050036, -0.00100050035, -0.00100050034, -0.00100050033])
        steps = np.exp([1, 2, 3, 4, 5])
        x0 = linear_initialization(metric, steps, bounds)
        npt.assert_almost_equal(x0[0], -(1 - (np.exp(-0.00100050038))), decimal=5)
        npt.assert_almost_equal(x0[1], 0, decimal=5)
        npt.assert_almost_equal(x0[2], 1, decimal=5)

    def test_linear_initialization_decreasing_unbounded(self):
        """
        Tests linear initialization for decreasing power law curve with c0 bounded [0,1]
        """
        metric = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        steps = np.array([20, 200, 2000, 2500, 3000])
        bounds = Constraints(scale=(None, None), negative_exponent=(0, None), asymptote=(0, 1))
        x0 = linear_initialization(metric, steps, bounds)
        assert x0[2] == 1.000  # applied bounds
        m_log = np.log((1.001) - metric)
        s_log = np.log(steps)
        m, b = linear_fit_manual(s_log, m_log)
        test = np.array([-np.exp(b), -m, 1.000])
        assert np.allclose(
            x0,
            test,
            rtol=0,
            atol=1e-8,
        )

    def test_linear_initialization_default(self, caplog):
        """
        Tests that with initialization failure, initial guess defaults to [0.5,0.5,1]
        """
        metric = np.array([0, 0.2, 0.4, 0.6, 0.8])
        steps = np.array([0, 200, 2000, 2500, 3000])
        bounds = Constraints(scale=(None, None), negative_exponent=(0, None), asymptote=(0, 1))
        err_msg = "Error applying linear initialization for initial guess, using default"
        with caplog.at_level(logging.WARNING):
            x0 = linear_initialization(metric, steps, bounds)
        assert err_msg in caplog.text
        assert all(x0 == np.array([0.5, 0.5, 1]))
