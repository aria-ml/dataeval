from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
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

        covariates, labels = mnist

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        metric = dp_metric()
        dataset_a = DamlDataset(even)
        dataset_b = DamlDataset(odd)
        result = metric.evaluate(dataset_a=dataset_a, dataset_b=dataset_b)
        assert result == output


@pytest.mark.parametrize(
    "dp_metric",
    [
        HP_FNN,
        HP_MST,
    ],
)
class TestDpCreateEncoding:
    """Tests that inputs can be encoded, not functionality"""

    @mock.patch("torch.vstack")
    def test_create_encoding(self, mock_vstack, dp_metric):
        model = MagicMock()
        empty_imgs = torch.Tensor([])
        m = dp_metric()
        m.model = model
        m._is_trained = True
        m.create_encoding(empty_imgs, empty_imgs)
        assert model.encode.call_count == 2
        assert mock_vstack.call_count == 1

    def test_create_encoding_no_model(self, dp_metric):
        m = dp_metric()
        m._is_trained = True

        x1 = x2 = torch.Tensor([])
        with pytest.raises(TypeError):
            m.create_encoding(x1, x2)

    def test_create_encoding_not_trained(self, dp_metric):
        m = dp_metric()
        x1 = x2 = torch.Tensor([])
        with pytest.raises(ValueError):
            m.create_encoding(x1, x2)


@pytest.mark.parametrize(
    "dp_metric",
    [
        HP_FNN,
        HP_MST,
    ],
)
class TestDpFitDataset:
    @mock.patch("daml._internal.metrics.aria.base.AERunner")
    def test_has_model(self, mock_runner, mnist, dp_metric):
        """Test given model is wrapped by AERunner"""

        class _TestModel(nn.Module):
            def forward(self, x):
                return x

        model = _TestModel()
        images, _ = mnist
        images = images[:, np.newaxis]
        dataset = DamlDataset(images)
        m = dp_metric()
        assert m.model is None
        m.fit_dataset(dataset, model=model)
        assert mock_runner.call_count == 1

    @mock.patch("daml._internal.metrics.aria.base.AETrainer")
    def test_no_model(self, mock_trainer, mnist, dp_metric):
        """Test default AETrainer is setup"""
        images, _ = mnist
        images = images[:, np.newaxis]
        dataset = DamlDataset(images)
        m = dp_metric()
        assert m.model is None
        m.fit_dataset(dataset)

        assert mock_trainer.call_count == 1

    def test_has_bad_model(self, mnist, dp_metric):
        images, _ = mnist
        images = images[:, np.newaxis]
        dataset = DamlDataset(images)
        model = "Not a Model"
        m = dp_metric()
        assert m.model is None
        with pytest.raises(TypeError):
            m.fit_dataset(dataset, model=model)  # type: ignore


@pytest.mark.parametrize(
    "dp_metric",
    [
        HP_FNN,
        HP_MST,
    ],
)
class TestDpEncode:
    def test_encode_default(self, dp_metric):
        """Default encode is False"""
        m = dp_metric()
        m.create_encoding = MagicMock()
        m.calculate_errors = MagicMock()
        x1 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        x2 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))

        assert m.encode is False
        m.evaluate(x1, x2)

        assert m.create_encoding.call_count == 0

    def test_encode_fit_dataset(self, dp_metric):
        """After fitting a dataset, encode is set to True"""
        m = dp_metric()
        m.create_encoding = MagicMock()
        m.calculate_errors = MagicMock()
        m.encode = True
        # Create 4-dim images for permute
        x1 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        x2 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))

        m.evaluate(x1, x2)

        assert m.create_encoding.call_count == 1

    def test_encode_override_true(self, dp_metric):
        """Override default encode with True breaks (no model fit)"""
        m = dp_metric()
        m.calculate_errors = MagicMock()
        # Create 4-dim images for permute
        x1 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        x2 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))

        assert m.encode is False
        assert m.model is None
        assert m._is_trained is False
        with pytest.raises(ValueError):
            m.evaluate(x1, x2, True)

    def test_encode_override_false(self, dp_metric):
        """Override encode with False after fitting dataset"""
        m = dp_metric()
        m.create_encoding = MagicMock()
        m.calculate_errors = MagicMock()
        m.encode = True
        x1 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))
        x2 = DamlDataset(np.ones(shape=(1, 1, 1, 1)))

        m.evaluate(x1, x2, False)

        assert m.create_encoding.call_count == 0


class TestDivergenceOutput:
    def test_divergenceoutput_eq(self):
        assert DivergenceOutput(1.0, 1.0) == DivergenceOutput(1.0, 1.0)

    def test_divergenceoutput_ne(self):
        assert DivergenceOutput(1.0, 1.0) != DivergenceOutput(0.9, 0.9)

    def test_divergenceoutput_ne_type(self):
        assert DivergenceOutput(1.0, 1.0) != (1.0, 1.0)
