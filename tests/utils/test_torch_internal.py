from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dataeval.config import get_device
from dataeval.utils.torch._internal import predict_batch, trainer
from dataeval.utils.torch.models import AE, ResNet18

model = AE((1, 16, 16))


@pytest.mark.required
@pytest.mark.parametrize("y_train", (None, np.zeros((3, 1, 16, 16))))
@pytest.mark.parametrize("loss_fn", (None, torch.nn.MSELoss()))
@pytest.mark.parametrize("optimizer", (None, torch.optim.Adam(model.parameters())))
@pytest.mark.parametrize("preprocess_fn", (None, torch.nn.Identity()))
@pytest.mark.parametrize("batch_size", (1, 3))
class TestTorchTrainerCPU:
    def test_torch_trainer(self, y_train, loss_fn, optimizer, preprocess_fn, batch_size):
        trainer(
            model=AE((1, 16, 16)),
            x_train=np.ones((3, 1, 16, 16)),
            y_train=y_train,
            loss_fn=loss_fn,
            optimizer=optimizer,
            preprocess_fn=preprocess_fn,
            epochs=1,
            batch_size=batch_size,
            device=torch.device("cpu"),
            verbose=False,
        )


@pytest.mark.optional
@pytest.mark.cuda
@pytest.mark.parametrize("y_train", (None, np.zeros((3, 1, 16, 16))))
@pytest.mark.parametrize("loss_fn", (None, torch.nn.MSELoss()))
@pytest.mark.parametrize("optimizer", (None, torch.optim.Adam(model.parameters())))
@pytest.mark.parametrize("preprocess_fn", (None, torch.nn.Identity()))
@pytest.mark.parametrize("batch_size", (1, 3))
class TestTorchTrainerCUDA:
    def test_torch_trainer(self, y_train, loss_fn, optimizer, preprocess_fn, batch_size):
        trainer(
            model=AE((1, 16, 16)),
            x_train=np.ones((3, 1, 16, 16)),
            y_train=y_train,
            loss_fn=loss_fn,
            optimizer=optimizer,
            preprocess_fn=preprocess_fn,
            epochs=1,
            batch_size=batch_size,
            device=torch.device("cuda"),
            verbose=False,
        )


@pytest.mark.required
class TestPredictBatch:
    n, n_features, n_classes, latent_dim = 100, 10, 5, 2
    x = np.zeros((n, n_features), dtype=np.float32)
    t = torch.zeros((n, n_features), dtype=torch.float32)

    class MyModel(torch.nn.Module):
        n_features, n_classes = 10, 5

        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(self.n_features, self.n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.dense(x)
            return out

    AutoEncoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dim), torch.nn.Linear(latent_dim, n_features))

    # model, batch size, preprocessing function
    tests_predict = [
        (x, AutoEncoder, 2, None),
        (x, AutoEncoder, int(1e10), None),
        (t, AutoEncoder, int(1e10), None),
        (x, AutoEncoder, int(1e10), lambda x: x),
        (t, AutoEncoder, int(1e10), lambda x: x),
        (x, lambda x: x, 2, None),
        (t, lambda x: x, 2, None),
        (x, lambda x: x, 2, lambda x: x),
        (t, lambda x: x, 2, lambda x: x),
    ]
    n_tests = len(tests_predict)

    @pytest.fixture(scope="class")
    def params(self, request):
        return self.tests_predict[request.param]

    @pytest.mark.parametrize("params", list(range(n_tests)), indirect=True)
    def test_predict_batch(self, params):
        x, model, batch_size, preprocess_fn = params
        preds = predict_batch(
            x,
            model,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            device=get_device("cpu"),
        )
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(model, torch.nn.Sequential) or hasattr(model, "__name__") and model.__name__ == "id_fn":
            assert preds.shape == self.x.shape
        elif isinstance(model, torch.nn.Module):
            assert preds.shape == (self.n, self.n_classes)

    def test_predict_batch_unsupported_model(self):
        with pytest.raises(TypeError):
            predict_batch(self.x, self.MyModel("unsupported"), device=get_device("cpu"))  # type: ignore


class TestResNet18:
    @patch("dataeval.utils.torch.models.resnet18", return_value=MagicMock())
    def test_resnet18_forward(self, mock_resnet18):
        model = ResNet18()
        assert mock_resnet18.call_count == 1
        model.model.return_value = "bar"  # type: ignore
        bar = model.forward("foo")  # type: ignore
        assert bar == "bar"

    @patch("dataeval.utils.torch.models.ResNet18_Weights")
    def test_resnet18_transforms(self, mock_weights):
        ResNet18.transforms()
        assert mock_weights.DEFAULT.transforms.call_count == 1

    @patch("dataeval.utils.torch.models.resnet18", return_value=MagicMock())
    def test_resnet18_str(self, mock_resnet18):
        model = ResNet18()
        assert "MagicMock" in str(model)
