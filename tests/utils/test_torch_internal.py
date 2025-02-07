import numpy as np
import pytest
import torch

from dataeval.utils.torch._internal import trainer
from dataeval.utils.torch.models import AE

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
@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="CUDA is not available")
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
