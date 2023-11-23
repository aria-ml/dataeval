import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from daml.datasets import DamlDataset
from daml.metrics.sufficiency import Sufficiency

np.random.seed(0)
torch.manual_seed(0)


def load_dataset():
    # Loads mnist dataset from binary
    path = "tests/datasets/mnist.npz"
    with np.load(path, allow_pickle=True) as fp:
        images, labels = fp["x_train"][:1000], fp["y_train"][:1000]
        test_images, test_labels = fp["x_test"][:100], fp["y_test"][:100]
    images = images.reshape((1000, 1, 28, 28))
    test_images = test_images.reshape((100, 1, 28, 28))
    train_ds = DamlDataset(images, labels)
    test_ds = DamlDataset(test_images, test_labels)
    return train_ds, test_ds


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


def custom_train(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
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
    epochs = 1

    for _ in range(epochs):
        # Zero out gradients
        optimizer.zero_grad()
        # Forward Propagation
        outputs = model(X)
        # Back prop
        loss = criterion(outputs, y)
        loss.backward()
        # Update optimizer
        optimizer.step()


def custom_eval(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
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
    with torch.no_grad():
        preds = model(X)
        result = metric(preds, y)
    return result


class TestClassificationSufficiency:
    train_ds, test_ds = load_dataset()

    def test_output_values(self):
        model = Net()
        length = len(self.train_ds)

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined training and eval functions
        suff.set_training_func(custom_train)
        suff.set_eval_func(custom_eval)
        # Create data indices for training
        m_count = 1
        num_steps = 3
        suff.setup(length, m_count, num_steps)
        # Train & test model
        output = suff.run(model, self.train_ds, self.test_ds)

        for k, v in output.items():
            print(f"{k}:{v}")

        # Loose params testing. TODO -> Find way to make tighter
        params = output["params"]
        assert params[0] >= 0.0
        assert params[1] >= 0.0
        assert params[2] >= 0.0
        assert params[2] <= 0.5

        params = output["params"]
        assert len(params) == 3

        # Accuracy should be bounded
        accuracy = output["metric"]
        assert np.all(0 <= accuracy)
        assert np.all(accuracy <= 1)

        # Geomshape should calculate deterministically
        geomshape = np.array(output["geomshape"])
        geomshape_answer = np.array(
            [int(0.01 * len(self.train_ds)), len(self.train_ds), num_steps]
        )
        npt.assert_array_equal(geomshape, geomshape_answer)

        # n_i test
        n_i = output["n_i"]
        npt.assert_array_equal(n_i, np.geomspace(*geomshape).astype(int))

        # p_i test
        p_i = output["p_i"]
        npt.assert_array_equal(p_i, 1 - np.mean(accuracy, axis=1))

    def test_setup_non_func(self):
        suff = Sufficiency()

        # Add more types: Scalars, Arrays, etc
        with pytest.raises(TypeError):
            suff.set_eval_func(None)

        with pytest.raises(TypeError):
            suff.set_training_func(None)

    def test_plot(self):
        # Only needed for plotting test
        import os

        suff = Sufficiency()

        output = {
            "metric": np.ones(shape=(3, 1)),
            "params": np.ones(shape=(3,)),
            "n_i": np.ones(shape=(3,)),
            "p_i": np.ones(shape=(3,)),
            "geomshape": (1, 100, 3),
        }

        suff.plot(output_dict=output)
        # Can only confirm file is created, not data in it
        assert os.path.exists("Sufficiency Plot.png")
        os.remove("Sufficiency Plot.png")

    def test_plot_empty_output(self):
        suff = Sufficiency()
        fake_output = {}

        # Tests empty params
        with pytest.raises(KeyError):
            suff.plot(fake_output)

        # Tests empty n_i, must fill params first
        fake_output["params"] = [0.24500099, 0.02901963, 0.64048776]
        with pytest.raises(KeyError):
            suff.plot(fake_output)

        # Tests empty p_i
        fake_output["n_i"] = [1, 10, 100]
        with pytest.raises(KeyError):
            suff.plot(fake_output)

        # Tests empty geomshape
        fake_output["p_i"] = [0.89, 0.86, 0.86]
        with pytest.raises(KeyError):
            suff.plot(fake_output)
