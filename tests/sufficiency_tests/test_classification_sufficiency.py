from copy import deepcopy

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
    # Loads dataset
    path = "tests/datasets/mnist.npz"
    with np.load(path, allow_pickle=True) as fp:
        images, labels = fp["x_train"][:3000], fp["y_train"][:3000]
        test_images, test_labels = fp["x_test"][:500], fp["y_test"][:500]
    images = images.reshape((3000, 1, 28, 28))
    test_images = test_images.reshape((500, 1, 28, 28))
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


class TestSufficiency:
    train_ds, test_ds = load_dataset()

    def test_f_out(self):
        n_i = np.geomspace(30, 3000, 20).astype(int)
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

        output = Sufficiency.f_out(n_i, x)

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
        n_i = np.geomspace(30, 3000, 20).astype(int)
        answer = [3.263266, 0.416022, 0.005313]

        output = Sufficiency.get_params(p_i, n_i)

        npt.assert_almost_equal(output, answer, decimal=5)

    def test_create_data_indices(self):
        N = 5
        M = 100

        output = Sufficiency._create_data_indices(N, M)

        assert output.shape == (N, M)

    def test_reset_parameters(self):
        """
        Test that resetting the parameters brings us back to the original model
        """

        # Setup model
        model = Net()
        # Get original weights
        state_dict = deepcopy(model.state_dict())

        # Reset the parameters
        model = Sufficiency.reset_parameters(model)
        reset_state_dict = model.state_dict()

        # Confirm reset_parameters changed the original parameters
        assert str(state_dict) != str(reset_state_dict)

    @pytest.mark.functional
    def test_run_output(self):
        model = Net()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        suff = Sufficiency(model, optimizer, criterion, acc)
        output = suff.run(self.train_ds, self.test_ds)

        answer = np.array([2.97047, 0.53136, 0.3346])

        # Test
        params = output["params"]
        npt.assert_almost_equal(answer, params, decimal=5)

    def test_outputs(self):
        """
        Testing that the outputs come back with the correct shape and type, not value
        """
        model = Net()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        suff = Sufficiency(model, optimizer, criterion, acc)
        output = suff.run(self.train_ds, self.test_ds)

        # Param tests
        params = output["params"]
        assert len(params) == 3

        # Accuracy tests
        accuracy = output["accuracy"]
        assert np.all(0 <= accuracy)
        assert np.all(accuracy <= 1)

        # Geomshape tests
        geomshape = np.array(output["geomshape"])
        geomshape_answer = np.array(
            [int(0.01 * len(self.train_ds)), len(self.train_ds), 20]
        )
        npt.assert_array_equal(geomshape, geomshape_answer)

        # n_i tests
        n_i = output["n_i"]
        npt.assert_array_equal(n_i, np.geomspace(*geomshape).astype(int))

        # p_i tests
        p_i = output["p_i"]
        npt.assert_array_equal(p_i, 1 - np.mean(accuracy, axis=1))

    def test_plotting(self):
        model = Net()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        suff = Sufficiency(model, optimizer, criterion, acc)
        suff.run(self.train_ds, self.test_ds, plot=True)

        import os

        assert os.path.exists("Sufficiency Plot.png")
