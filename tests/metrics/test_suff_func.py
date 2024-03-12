from typing import Dict

import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, Subset

from daml._internal.metrics.sufficiency import STEPS_KEY
from daml.metrics import Sufficiency
from tests.utils.data import DamlDataset


class Net(nn.Module):
    def __init__(self) -> None:
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


def custom_train(model, dataset, indices):
    """
    Passes data once through the model with backpropagation

    Parameters
    ----------
    model : nn.Module
        The trained model that will be evaluated
    X : torch.Tensor
        The training data to be passed through the model
    y : torch.Tensor
        The training labels corresponding to the data
    """
    # Defined only for this testing scenario
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for X, y in DataLoader(Subset(dataset, indices)):
        # Zero out gradients
        optimizer.zero_grad()
        # Forward Propagation
        outputs = model(X)
        # Back prop
        loss = criterion(outputs, y)
        loss.backward()
        # Update optimizer
        optimizer.step()


def custom_eval(model, dataset) -> Dict[str, float]:
    """
    Evaluate a model on a single pass with a given metric

    Parameters
    ----------
    model : nn.Module
        The trained model that will be evaluated
    X : torch.Tensor
        The testing data to be passed through th model
    y : torch.Tensor
        The testing labels corresponding to the data

    Returns
    -------
    float
        The calculated performance of the model
    """
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    # Set model layers into evaluation mode
    model.eval()
    # Tell PyTorch to not track gradients, greatly speeds up processing
    result: float = 0.0
    for X, y in DataLoader(dataset):
        with torch.no_grad():
            preds = model(X)
            result = metric(preds, y)
    return {"Accuracy": result}


# @pytest.mark.functional
class TestSufficiencyFunctional:
    def test_classification(self, mnist) -> None:
        model = Net()
        length = 1000
        train_ds = DamlDataset(*mnist(length, "train", np.float32, "channels_first"))
        test_ds = DamlDataset(*mnist(100, "test", np.float32, "channels_first"))
        m_count = 1
        steps = 3

        # Instantiate sufficiency metric
        suff = Sufficiency(
            model,
            train_ds,
            test_ds,
            custom_train,
            custom_eval,
            m_count,
            steps,
        )

        # Train & test model
        output = suff.evaluate()

        # Accuracy should be bounded
        accuracy = output["Accuracy"]
        assert np.all(0 <= accuracy)
        assert np.all(accuracy <= 1)
        assert len(accuracy) == 3

        # Geomshape should calculate deterministically
        geomshape = output[STEPS_KEY]
        geomshape_answer = np.geomspace(0.01 * length, length, steps).astype(np.int64)
        npt.assert_array_equal(geomshape, geomshape_answer)
