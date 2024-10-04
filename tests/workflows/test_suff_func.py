from __future__ import annotations

from typing import Sequence, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, Dataset, Subset

from dataeval._internal.workflows.sufficiency import SufficiencyOutput
from dataeval.workflows import Sufficiency
from tests.conftest import mnist
from tests.utils.data import DataEvalDataset


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


def mock_ds(length: int | None):
    ds = MagicMock()
    if length is None:
        delattr(ds, "__len__")
    else:
        ds.__len__.return_value = length
    return ds


def realistic_train(model: nn.Module, dataset: Dataset, indices: Sequence[int]):
    # Defined only for this testing scenario
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 100  # 10

    # Define the dataloader for training
    dataloader = DataLoader(Subset(dataset, indices), batch_size=16)

    for epoch in range(epochs):
        for batch in dataloader:
            # Load data/images to device
            X = torch.Tensor(batch[0]).to(device)
            # Load targets/labels to device
            y = torch.Tensor(batch[1]).to(device)
            # Zero out gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(X)
            # Compute loss
            loss = criterion(outputs, y)
            # Back prop
            loss.backward()
            # Update weights/parameters
            optimizer.step()


def realistic_eval(model: nn.Module, dataset: Dataset) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    result = 0

    # Set model layers into evaluation mode
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16)
    # Tell PyTorch to not track gradients, greatly speeds up processing
    with torch.no_grad():
        for batch in dataloader:
            # Load data/images to device
            X = torch.Tensor(batch[0]).to(device)
            # Load targets/labels to device
            y = torch.Tensor(batch[1]).to(device)
            preds = model(X)
            metric.update(preds, y)
        result = metric.compute().cpu()
    return {"Accuracy": result}


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


def custom_eval(model, dataset) -> dict[str, float]:
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


class TestSufficiencyFunctional:
    def test_classification(self) -> None:
        model = Net()
        length = 1000
        train_ds = DataEvalDataset(*mnist(train=True, size=length, dtype=np.float32, channels="channels_first"))
        test_ds = DataEvalDataset(*mnist(train=False, size=100, dtype=np.float32, channels="channels_first"))
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
        output = suff.evaluate(niter=100)

        # Accuracy should be bounded
        accuracy = cast(np.ndarray, output.measures["Accuracy"])
        assert np.all(accuracy >= 0)
        assert np.all(accuracy <= 1)
        assert len(accuracy) == 3

        # Geomshape should calculate deterministically
        geomshape = cast(np.ndarray, output.steps)
        geomshape_answer = np.geomspace(0.01 * length, length, steps).astype(np.int64)
        npt.assert_array_equal(geomshape, geomshape_answer)


class TestSufficiencyInverseProjectFunc:
    def test_predicts_on_real_data(self):
        """
        End-to-end functional test of sufficiency. This loads the MNIST dataset,
        fits a sufficiency curve to it, and then predicts how many steps are required
        to achieve various levels of model accuracy. The test passes if the accuracy
        values of the model at the predicted steps is within 0.5 of the desired
        accuracies.
        """
        np.random.seed(0)
        torch.manual_seed(0)
        torch.set_float32_matmul_precision("high")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch._dynamo.config.suppress_errors = True  # type: ignore

        train_ds = DataEvalDataset(
            *mnist(train=True, size=1000, unit_normalize=True, dtype=np.float32, channels="channels_first")
        )
        test_ds = DataEvalDataset(
            *mnist(train=False, size=200, unit_normalize=True, dtype=np.float32, channels="channels_first")
        )
        model = cast(Net, torch.compile(Net().to(device)))

        # Instantiate sufficiency metric
        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=realistic_train,
            eval_fn=realistic_eval,
            runs=2,
            substeps=10,
        )

        # Initialize the array of accuracies that we want to achieve
        desired_accuracies = {"Accuracy": np.array([0.4, 0.6, 0.8])}

        """
        Normally we would write output_to_fit = suff.evaluate()
        However, this takes a very long time to evaluate, so for this test,
        the output from suff.evaluate() is pasted below.
        """
        output_to_fit = SufficiencyOutput(
            steps=np.array([10, 16, 27, 46, 77, 129, 215, 359, 599, 1000]),
            params={"Accuracy": np.array([-0.32791097, -0.27294133, 1.16538843])},
            measures={"Accuracy": np.array([0.345, 0.24, 0.2975, 0.32, 0.335, 0.4, 0.8275, 0.855, 0.8725, 0.9075])},
        )

        # Evaluate the learning curve to infer the needed amount of training data
        # to train a model to (desired_accuracies) accuracy
        pred_nsamples = Sufficiency.inv_project(desired_accuracies, output_to_fit)["Accuracy"]

        # Train model and see if we get the accuracy we expect on these predicted
        # amounts of training data
        output = suff.evaluate(pred_nsamples, niter=600)
        proj_accuracies = cast(np.ndarray, output.measures["Accuracy"])
        npt.assert_allclose(proj_accuracies, desired_accuracies["Accuracy"], rtol=0.1, atol=1)
