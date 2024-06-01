from typing import Dict, Optional, Sequence, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, Dataset, Subset

import daml._internal.metrics.sufficiency as dms
from daml._internal.metrics.sufficiency import PARAMS_KEY, STEPS_KEY
from daml.metrics import Sufficiency
from tests.utils.data import DamlDataset


class MockNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, x):
        return x

    def forward(self, x):
        pass


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


def mock_ds(length: Optional[int]):
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


def realistic_eval(model: nn.Module, dataset: Dataset) -> Dict[str, float]:
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
        output = suff.evaluate(niter=100)

        # Accuracy should be bounded
        accuracy = cast(np.ndarray, output["Accuracy"])
        assert np.all(accuracy >= 0)
        assert np.all(accuracy <= 1)
        assert len(accuracy) == 3

        # Geomshape should calculate deterministically
        geomshape = cast(np.ndarray, output[STEPS_KEY])
        geomshape_answer = np.geomspace(0.01 * length, length, steps).astype(np.int64)
        npt.assert_array_equal(geomshape, geomshape_answer)


class TestSufficiencyExtraFeaturesFunc:
    def test_f_inv_out(self):
        """
        Tests that f_inv_out exactly inverts f_out.
        """

        n_i = np.array([1.234])
        x = np.array([1.1, 2.2, 3.3])
        # Predict y from n_i evaluated on curve defined by x
        y = dms.f_out(n_i, x)
        # Feed y into inverse function to get the original n_i back out
        n_i_recovered = dms.f_inv_out(y, x)

        assert np.isclose(n_i[0], n_i_recovered[0])

    def test_inv_project_steps(self):
        """
        Verifies that inv_project_steps is the inverse of project_steps
        """
        measure = np.array([1, 2, 3])
        steps = np.array([4, 5, 6])
        projection = np.array([7, 8, 9])

        params = dms.calc_params(p_i=(1 - measure), n_i=steps, niter=1000)
        accuracies = dms.project_steps(params, projection)
        predicted_proj = dms.inv_project_steps(params, accuracies)

        assert np.all(np.isclose(projection, predicted_proj, atol=1))

    def test_can_invert_sufficiency(self):
        num_samples = np.arange(20, 80, step=10)
        accuracies = num_samples / 100

        params = dms.calc_params(1 - accuracies, num_samples, 1000)

        data = {STEPS_KEY: num_samples, PARAMS_KEY: {"Accuracy": params}, "Accuracy": accuracies}

        desired_accuracies = {"Accuracy": np.array([0.4, 0.6])}
        needed_data = Sufficiency.inv_project(desired_accuracies, data)["Accuracy"]

        target_needed_data = np.array([40, 60])
        assert np.all(np.isclose(needed_data, target_needed_data, atol=1))

    def test_predicts_on_real_data(self, mnist):
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

        train_ds = DamlDataset(*mnist(1000, "train", np.float32, "channels_first", True))
        test_ds = DamlDataset(*mnist(200, "test", np.float32, "channels_first", True))
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
        output_to_fit = {
            STEPS_KEY: np.array([10, 16, 27, 46, 77, 129, 215, 359, 599, 1000]),
            PARAMS_KEY: {"Accuracy": np.array([-0.32791097, -0.27294133, 1.16538843])},
            "Accuracy": np.array([0.345, 0.24, 0.2975, 0.32, 0.335, 0.4, 0.8275, 0.855, 0.8725, 0.9075]),
        }

        # Evaluate the learning curve to infer the needed amount of training data
        # to train a model to (desired_accuracies) accuracy
        pred_nsamples = Sufficiency.inv_project(desired_accuracies, output_to_fit)["Accuracy"]

        # Train model and see if we get the accuracy we expect on these predicted
        # amounts of training data
        output = suff.evaluate(pred_nsamples, niter=600)
        proj_accuracies = cast(np.ndarray, output["Accuracy"])
        assert np.all(np.isclose(proj_accuracies, desired_accuracies["Accuracy"], atol=1))
