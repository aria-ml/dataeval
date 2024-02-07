import numbers
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from daml.metrics.sufficiency import Sufficiency
from tests.utils.data import DamlDataset

np.random.seed(0)
torch.manual_seed(0)


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


def train_task_cls(model: nn.Module, dl: DataLoader):
    # Extract first batch in dataloader
    batch = next(iter(dl))
    assert len(batch) == 2

    img, lbl = batch
    # Each image has a label
    assert img.shape[0] == lbl.shape[0]


def train_task_cls_kwargs(model: nn.Module, dl: DataLoader, num: int) -> None:
    assert isinstance(num, int)


def train_task_od(model: nn.Module, dl: DataLoader):
    # Extract first batch from dataloader
    batch = next(iter(dl))
    assert len(batch)

    img, lbls, bxs = batch

    # Each image has a set of labels and boxes
    assert img.shape[0] == lbls.shape[0] == bxs.shape[0]
    # Each box as a label
    assert lbls.shape[1] == bxs.shape[1]


def eval_100(model: nn.Module, dl: DataLoader):
    """Eval should always return a float, and error if not"""
    return 1.0


def eval_kwargs(model: nn.Module, dl: DataLoader, num: int):
    """Kwargs should match parameter input"""
    assert isinstance(num, int)
    return num


class TestSufficiency:
    def test_mock_run(self) -> None:
        suff = Sufficiency()

        suff._train = MagicMock()
        suff._eval = MagicMock()
        suff.setup(100, 1, 2)

        patch("torch.utils.data.DataLoader").start()

        model = MagicMock()
        train_ds = MagicMock()
        test_ds = MagicMock()

        results = suff.run(model, train_ds=train_ds, test_ds=test_ds)

        assert isinstance(results, dict)

    def test_mock_run_with_kwargs(self) -> None:
        suff = Sufficiency()

        suff._train = MagicMock()
        suff._eval = MagicMock()
        suff.setup(100, 1, 2)

        patch("torch.utils.data.DataLoader").start()

        model = MagicMock()
        train_ds = MagicMock()
        test_ds = MagicMock()

        results = suff.run(
            model,
            train_ds=train_ds,
            test_ds=test_ds,
            train_kwargs={"train": 1},
            eval_kwargs={"eval": 1},
        )

        assert suff._train.call_count == 2
        assert {"train": 1} in suff._train.call_args[0]

        assert suff._eval.call_count == 2
        assert {"eval": 1} in suff._eval.call_args[0]

        assert isinstance(results, dict)

    def test_train_func_is_none(self) -> None:
        suff = Sufficiency()

        model = MagicMock()
        train_ds = MagicMock()

        with pytest.raises(TypeError):
            suff._train(model, train_ds, {})

    def test_eval_func_is_none(self) -> None:
        suff = Sufficiency()

        model = MagicMock()
        test_ds = MagicMock()

        with pytest.raises(TypeError):
            suff._eval(model, test_ds, {})

    def test_train_kwargs(self, mock_net) -> None:
        """Tests correct kwarg handling"""
        train_ds, _ = load_cls_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined training and eval functions
        suff.set_training_func(train_task_cls_kwargs)
        dl = DataLoader(train_ds)

        kwargs = {"num": 1}
        suff._train(mock_net, dataloader=dl, kwargs=kwargs)

    def test_eval_kwargs(self, mock_net) -> None:
        """Tests kwarg handling"""
        _, test_ds = load_cls_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined eval function
        suff.set_eval_func(eval_kwargs)
        dl = DataLoader(test_ds)

        kwargs = {"num": 100}
        result = suff._eval(mock_net, dataloader=dl, kwargs=kwargs)
        assert result == kwargs["num"]

    def test_setup_non_func(self) -> None:
        """Tests if the set function is robust against invalid types"""
        suff = Sufficiency()

        # for t in [None, 1, 1.0, "func", {"func": 0}]:  # Hypothesis testing, errors
        with pytest.raises(TypeError):
            suff.set_eval_func(None)  # type: ignore

        with pytest.raises(TypeError):
            suff.set_training_func(None)  # type: ignore

    def test_setup(self) -> None:
        """Tests that setup correctly sets internal variables"""
        suff = Sufficiency()

        LENGTH = 100
        COUNT = 3
        SUBSTEPS = 10

        # Sets _outputs, _geomshape, _ranges, and _indices
        suff.setup(length=LENGTH, num_models=COUNT, substeps=SUBSTEPS)

        output_answer = np.zeros((SUBSTEPS, COUNT))
        npt.assert_array_equal(output_answer, suff._outputs)

        geomshape_answer = (0.01 * LENGTH, LENGTH, SUBSTEPS)
        npt.assert_array_equal(geomshape_answer, suff._geomshape)

        ranges_answer = np.geomspace(int(0.01 * LENGTH), LENGTH, SUBSTEPS).astype(int)

        npt.assert_array_equal(ranges_answer, suff._ranges)

        indices_answer_shape = (COUNT, LENGTH)
        assert indices_answer_shape == suff._indices.shape

    def test_invalid_setup(self) -> None:
        suff = Sufficiency()

        # Invalid length
        with pytest.raises(ValueError):
            suff.setup(0, 1, 1)

        with pytest.raises(ValueError):
            suff.setup(-1, 1, 1)

    def test_plot(self):
        """Tests that a plot is generated and saved"""
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

    def test_plot_missing_output_keys(self):
        """Tests that custom dictionaries have all keys"""
        suff = Sufficiency()
        fake_output = {}

        # Missing key: params
        with pytest.raises(KeyError):
            suff.plot(fake_output)

        # Missing key: n_i
        fake_output["params"] = [0.24500099, 0.02901963, 0.64048776]
        with pytest.raises(KeyError):
            suff.plot(fake_output)

        # Missing key: p_i
        fake_output["n_i"] = [1, 10, 100]
        with pytest.raises(KeyError):
            suff.plot(fake_output)

        # Missing key: geomshape
        fake_output["p_i"] = [0.89, 0.86, 0.86]
        with pytest.raises(KeyError):
            suff.plot(fake_output)


class TestSufficiencyCls:
    def test_train(self, mock_net) -> None:
        train_ds, _ = load_cls_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined training and eval functions
        suff.set_training_func(train_task_cls)

        trainloader = DataLoader(train_ds)
        suff._train(mock_net, trainloader, {})

    def test_eval_result(self, mock_net) -> None:
        _, test_ds = load_cls_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined training and eval functions
        suff.set_eval_func(eval_100)

        testloader = DataLoader(test_ds)
        result = suff._eval(mock_net, testloader, {})

        # Result is a number (int, float) not Iterable, str, etc
        assert isinstance(result, numbers.Real)


# Can be combined with classification using parameterize, not sure if worth
class TestSufficiencyOD:
    def test_train(self, mock_net) -> None:
        train_ds, _ = load_od_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined training and eval functions
        suff.set_training_func(train_task_od)

        trainloader = DataLoader(train_ds)
        suff._train(mock_net, trainloader, {})

    def test_eval_result(self, mock_net) -> None:
        _, test_ds = load_od_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined training and eval functions
        suff.set_eval_func(eval_100)

        testloader = DataLoader(test_ds)
        result = suff._eval(mock_net, testloader, {})

        # Result is a number (int, float) not Iterable, str, etc
        assert isinstance(result, numbers.Real)
