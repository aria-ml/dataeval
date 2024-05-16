from typing import Dict, Optional, Sequence, Tuple, cast
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset, Subset

import daml._internal.metrics.sufficiency as dms
from daml._internal.metrics.sufficiency import STEPS_KEY
from daml.metrics import Sufficiency
from tests.utils.data import DamlDataset

np.random.seed(0)
torch.manual_seed(0)


class MockNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, x):
        return x

    def forward(self, x):
        pass


class RealisticNet(nn.Module):
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


def load_cls_dataset() -> Tuple[DamlDataset, DamlDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))

    train_ds = DamlDataset(images, labels)
    test_ds = DamlDataset(images, labels)

    return train_ds, test_ds


def load_od_dataset() -> Tuple[DamlDataset, DamlDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))
    boxes = np.ones(shape=(1, 1, 4))

    train_ds = DamlDataset(images, labels, boxes)
    test_ds = DamlDataset(images, labels, boxes)

    return train_ds, test_ds


def eval_100(model: nn.Module, dl: DataLoader) -> Dict[str, float]:
    """Eval should always return a float, and error if not"""
    return {"eval": 1.0}


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
        result = metric.compute()
    return {"Accuracy": result}


class TestSufficiency:
    def test_mock_run(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"test": 1.0}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        results = suff.evaluate()
        assert isinstance(results, dict)

    def test_mock_run_with_kwargs(self) -> None:
        train_fn = MagicMock()
        eval_fn = MagicMock()
        eval_fn.return_value = {"test": 1.0}
        train_kwargs = {"train": 1}
        eval_kwargs = {"eval": 1}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=train_fn,
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
            train_kwargs=train_kwargs,
            eval_kwargs=eval_kwargs,
        )

        results = suff.evaluate()

        assert train_fn.call_count == 2
        assert train_kwargs == train_fn.call_args.kwargs

        assert eval_fn.call_count == 2
        assert eval_kwargs == eval_fn.call_args.kwargs

        assert isinstance(results, dict)

    def test_run_with_invalid_key(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {STEPS_KEY: 1.0}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        with pytest.raises(KeyError):
            suff.evaluate()

    @pytest.mark.parametrize(
        "train_ds_len, test_ds_len, expected_error",
        [
            (None, 1, TypeError),
            (1, None, TypeError),
            (0, 1, ValueError),
            (1, 0, ValueError),
            (1, 1, None),
        ],
    )
    def test_dataset_len(self, train_ds_len, test_ds_len, expected_error):
        def call_suff(train_ds_len, test_ds_len):
            Sufficiency(
                model=MagicMock(),
                train_ds=mock_ds(train_ds_len),
                test_ds=mock_ds(test_ds_len),
                train_fn=MagicMock(),
                eval_fn=MagicMock(),
            )

        if expected_error is None:
            call_suff(train_ds_len, test_ds_len)
            return

        with pytest.raises(expected_error):
            call_suff(train_ds_len, test_ds_len)

    def test_train_fn_is_non_callable(self):
        with pytest.raises(TypeError):
            Sufficiency(
                model=MagicMock(),
                train_ds=mock_ds(1),
                test_ds=mock_ds(1),
                train_fn=NonCallableMagicMock(),
                eval_fn=MagicMock(),
            )

    def test_eval_fn_is_non_callable(self):
        with pytest.raises(TypeError):
            Sufficiency(
                model=MagicMock(),
                train_ds=mock_ds(1),
                test_ds=mock_ds(1),
                train_fn=MagicMock(),
                eval_fn=NonCallableMagicMock(),
            )


class TestSufficiencyPlot:
    def test_plot(self):
        """Tests that a plot is generated"""
        # Only needed for plotting test
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test": np.array([0.2, 0.6, 0.9]),
        }
        result = Sufficiency.plot(output)
        assert len(result) == 1
        assert isinstance(result[0], Figure)

    def test_multiplot(self):
        """Tests that the multiple plots are generated"""
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([0.2, 0.6, 0.9]),
            "test2": np.array([0.2, 0.6, 0.9]),
            "test3": np.array([0.2, 0.6, 0.9]),
        }

        result = Sufficiency.plot(output)
        assert len(result) == 3
        assert isinstance(result[0], Figure)

    def test_multiplot_classwise(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
        }

        result = Sufficiency.plot(output)
        assert len(result) == 2
        assert isinstance(result[0], Figure)

    def test_multiplot_classwise_invalid_names(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
        }

        with pytest.raises(IndexError):
            Sufficiency.plot(output, ["A", "B", "C"])

    def test_multiplot_classwise_with_names(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
        }

        result = Sufficiency.plot(output, ["A", "B"])
        assert result[0].axes[0].get_title().startswith("test1_A")

    def test_multiplot_classwise_without_names(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
        }

        result = Sufficiency.plot(output)
        assert result[0].axes[0].get_title().startswith("test1_0")

    def test_multiplot_mixed(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
            "test2": np.array([0.2, 0.6, 0.9]),
        }

        result = Sufficiency.plot(output)
        assert len(result) == 3
        assert result[0].axes[0].get_title().startswith("test1_0")
        assert result[1].axes[0].get_title().startswith("test1_1")
        assert result[2].axes[0].get_title().startswith("test2")


class TestSufficiencyProject:
    def test_no_steps_key(self):
        output = {"test1": np.array([0.2, 0.6, 0.9])}
        with pytest.raises(KeyError):
            Sufficiency.project(output, 10000)

    def test_measure_length_invalid(self):
        output = {
            STEPS_KEY: np.array([10, 100]),
            "test1": np.array([0.2, 0.6, 0.9]),
        }
        with pytest.raises(ValueError):
            Sufficiency.project(output, 10000)

    @pytest.mark.parametrize("steps", [100, [100], np.array([100])])
    def test_project(self, steps):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([0.2, 0.6, 0.9]),
        }
        result = Sufficiency.project(output, steps)
        npt.assert_almost_equal(result["test1"], [0.6], decimal=4)

    def test_project_invalid_steps(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([0.2, 0.6, 0.9]),
        }
        with pytest.raises(ValueError):
            Sufficiency.project(output, 1.0)  # type: ignore

    def test_project_classwise(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
        }

        result = Sufficiency.project(output, [1000, 2000, 4000, 8000])
        assert len(result.keys()) == 2
        assert result["test1"].shape == (4, 2)

    def test_project_mixed(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([[0.2, 0.3], [0.6, 0.4], [0.9, 0.8]]),
            "test2": np.array([0.2, 0.6, 0.9]),
        }

        result = Sufficiency.project(output, [1000, 2000, 4000, 8000])
        assert len(result.keys()) == 3
        assert result["test1"].shape == (4, 2)
        assert result["test2"].shape == (4,)


class TestSufficiencyExtraFeatures:
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
        Verifies that inv_project_steps is the inverse of project_steps (within 1%)
        """
        measure = np.array([1.1, 2.2, 3.3])
        steps = np.array([4.4, 5.5, 6.6])
        projection = np.array([7.7, 8.8, 9.9])

        accuracies, _ = dms.project_steps(measure, steps, projection)
        predicted_proj = dms.inv_project_steps(measure, steps, accuracies)

        percent_error = np.linalg.norm(projection - predicted_proj) / np.linalg.norm(projection) * 100

        assert percent_error < 0.01

    def test_cached_params(self):
        """
        Similar to the above test_inv_project_steps, except we use the cached
        parameters from project_steps for the inverse function, rather than
        re-doing the curve fit inside inv_project_steps.
        """
        # TODO: A good test would be to verify that inv_project_steps actually
        # uses the cached params rather than re-calculating them.
        measure = np.array([1.1, 2.2, 3.3])
        steps = np.array([4.4, 5.5, 6.6])
        projection = np.array([7.7, 8.8, 9.9])

        accuracies, params = dms.project_steps(measure, steps, projection)
        predicted_proj = dms.inv_project_steps(measure, steps, accuracies, params)

        percent_error = np.linalg.norm(projection - predicted_proj) / np.linalg.norm(projection) * 100

        assert percent_error < 0.01

    def test_can_invert_sufficiency(self):
        """
        This loads mock sufficiency data, fits a sufficiency curve to it,
        and then predicts how many steps are required to achieve various
        levels of model accuracy. The test passes if the accuracy values
        of the model at the predicted steps is within 0.05 of the desired accuracies.
        """
        num_samples = np.arange(1, 80, step=10)
        accuracies = num_samples / 100
        # num_samples being too long may take too many iters for calc_params to converge

        # Mock arguments to initialize a Sufficiency object
        eval_fn = MagicMock()
        eval_fn.return_value = {"test": 1.0}
        patch("torch.utils.data.DataLoader").start()

        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=eval_fn,
            runs=1,
            substeps=2,
        )

        data = {}
        data["_STEPS_"] = num_samples
        data["Accuracy"] = accuracies

        desired_accuracies = np.array([0.2, 0.4, 0.6])
        needed_data = suff.inv_project(desired_accuracies, data)

        target_needed_data = np.array([20, 40, 60])
        assert np.all(np.isclose(needed_data, target_needed_data, atol=1))

    @pytest.mark.functional
    def test_predicts_on_real_data(self):
        """
        End-to-end functional test of sufficiency. This loads the MNIST dataset,
        fits a sufficiency curve to it, and then predicts how many steps are required
        to achieve various levels of model accuracy. The test passes if the accuracy
        values of the model at the predicted steps is within 0.05 of the desired
        accuracies.
        """
        np.random.seed(0)
        np.set_printoptions(formatter={"float": lambda x: f"{x:0.4f}"})
        torch.manual_seed(0)
        torch.set_float32_matmul_precision("high")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch._dynamo.config.suppress_errors = True  # type: ignore

        
        datasets.MNIST("./data", train=True, download=True)
        datasets.MNIST("./data", train=False, download=True)

        # Download the mnist dataset and preview the images
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=to_tensor)
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=to_tensor)

        # Take a subset of 2000 training images and 500 test images
        train_ds = Subset(train_ds, range(4000))
        test_ds = Subset(test_ds, range(500))
        

        # Compile the model
        model = torch.compile(RealisticNet().to(device))

        # Type cast the model back to Net as torch.compile returns a Unknown
        # Nothing internally changes from the cast; we are simply signaling the type
        model = cast(RealisticNet, model)

        # Instantiate sufficiency metric
        suff = Sufficiency(
            model=model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_fn=realistic_train,
            eval_fn=realistic_eval,
            runs=5,
            substeps=10,
        )

        # Train & test model
        #output_to_fit = suff.evaluate()

        # Initialize the array of accuracies that we want to achieve
        desired_accuracies = np.array([0.5, 0.8, 0.9])

        output_to_fit = {'_STEPS_': np.array([  40,   66,  111,  185,  309,  516,  861, 1437, 2397, 4000]), 'Accuracy': np.array([0.5976, 0.6732, 0.7584, 0.8048, 0.8428, 0.8936, 0.9136, 0.9388,
        0.9448, 0.9644])}

        # Evaluate the learning curve to infer the needed amount of training data
        # to train a model to (desired_accuracies) accuracy
        pred_nsamples = Sufficiency.inv_project(desired_accuracies, output_to_fit)

        # Train model and see if we get the accuracy we expect on these predicte
        # amounts of training data
        output_on_pred_nsamples = suff.evaluate(pred_nsamples)
        assert np.all(np.isclose(output_on_pred_nsamples["Accuracy"], desired_accuracies, atol=0.05))
