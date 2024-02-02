from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch.nn as nn

from daml.datasets import DamlDataset
from daml.metrics.divergence import HP_FNN, HP_MST, DivergenceOutput

np.random.seed(0)


class TestDpDivergence:
    @pytest.mark.parametrize(
        "dp_metric, output",
        [
            (
                HP_MST,
                DivergenceOutput(
                    dpdivergence=0.8377897755491117,
                    error=81.0,
                ),
            ),
            (
                HP_FNN,
                DivergenceOutput(
                    dpdivergence=0.8618209199122062,
                    error=69.0,
                ),
            ),
        ],
    )
    def test_dp_divergence(self, mnist, dp_metric, output):
        """Unit testing of Dp Divergence

        TBD
        """

        covariates, labels = mnist(add_channels=True)

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        dataset_a = DamlDataset(even)
        dataset_b = DamlDataset(odd)
        metric = dp_metric(dataset_a, dataset_b)
        result = metric.evaluate()
        assert result == output


@pytest.mark.parametrize(
    "dp_metric",
    [
        HP_FNN,
        HP_MST,
    ],
)
class TestDpDivergenceFit:
    @mock.patch("daml._internal.metrics.aria.base.AERunner")
    def test_has_trained_model(self, mock_runner, mnist, dp_metric):
        """Test given model is wrapped by AERunner"""

        class _TestModel(nn.Module):
            def forward(self, x):
                return x

        model = _TestModel()
        images, _ = mnist(add_channels=True)
        dataset = DamlDataset(images)
        m = dp_metric(dataset, dataset, True, model, False)
        m._fit()
        assert mock_runner.call_count == 1

    @mock.patch("daml._internal.metrics.aria.base.AETrainer")
    def test_has_untrained_model(self, mock_trainer, mnist, dp_metric):
        """Test given model is wrapped by AERunner"""

        class _TestModel(nn.Module):
            def forward(self, x):
                return x

        model = _TestModel()
        images, _ = mnist(add_channels=True)
        dataset = DamlDataset(images)
        m = dp_metric(dataset, dataset, True, model, True, 10)
        m._fit()
        assert mock_trainer.call_count == 1

    @mock.patch("daml._internal.metrics.aria.base.AETrainer")
    def test_no_model(self, mock_trainer, mnist, dp_metric):
        """Test default AETrainer is setup"""
        images, _ = mnist(add_channels=True)
        dataset = DamlDataset(images)
        m = dp_metric(dataset, dataset)
        assert m.model is None
        m._fit()

        assert mock_trainer.call_count == 1

    def test_has_bad_model(self, mnist, dp_metric):
        images, _ = mnist(add_channels=True)
        dataset = DamlDataset(images)
        m = dp_metric(dataset, dataset)
        m.model = "Not a Model"
        with pytest.raises(TypeError):
            m._fit()


@pytest.mark.parametrize(
    "dp_metric",
    [
        HP_FNN,
        HP_MST,
    ],
)
class TestDpEncode:
    @mock.patch("numpy.vstack")
    def test_create_encoding(self, mock_vstack, dp_metric):
        empty_ds = DamlDataset(np.ndarray([]))
        m = dp_metric(empty_ds, empty_ds, True)
        m._encode = MagicMock()
        m._encode_and_vstack()

        assert m._encode.call_count == 2
        assert mock_vstack.call_count == 1

    def test_encode_default_false(self, dp_metric):
        """Default encode is False"""
        x1 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        x2 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        m = dp_metric(x1, x2)
        m._fit = MagicMock()
        m._encode_and_vstack = MagicMock()
        m.calculate_errors = MagicMock()

        assert m.encode is False
        m.evaluate()

        assert m._fit.call_count == 0
        assert m._encode_and_vstack.call_count == 1
        assert m.calculate_errors.call_count == 1

    def test_encode_override_true(self, dp_metric):
        """Override default encode with True"""
        # Create 4-dim images for permute
        x1 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        x2 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        m = dp_metric(x1, x2, True)
        m._fit = MagicMock()
        m._encode_and_vstack = MagicMock()
        m.calculate_errors = MagicMock()

        assert m.encode is True
        m.evaluate()

        assert m._fit.call_count == 1
        assert m._encode_and_vstack.call_count == 1
        assert m.calculate_errors.call_count == 1


class TestDivergenceOutput:
    def test_divergenceoutput_eq(self):
        assert DivergenceOutput(1.0, 1.0) == DivergenceOutput(1.0, 1.0)

    def test_divergenceoutput_ne(self):
        assert DivergenceOutput(1.0, 1.0) != DivergenceOutput(0.9, 0.9)

    def test_divergenceoutput_ne_type(self):
        assert DivergenceOutput(1.0, 1.0) != (1.0, 1.0)
