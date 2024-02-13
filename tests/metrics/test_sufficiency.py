from typing import Dict, Tuple
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from daml.metrics.sufficiency import Sufficiency, SufficiencyOutput
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


def eval_100(model: nn.Module, dl: DataLoader) -> Dict[str, float]:
    """Eval should always return a float, and error if not"""
    return {"eval": 1.0}


class TestSufficiency:
    def test_mock_run(self) -> None:
        suff = Sufficiency()

        suff._train = MagicMock()
        suff._eval = MagicMock()
        suff._eval.return_value = {"test": 1.0}

        patch("torch.utils.data.DataLoader").start()

        model = MagicMock()
        train_ds = MagicMock()
        train_ds.__len__.return_value = 2
        test_ds = MagicMock()

        results = suff.run(
            model,
            train_ds=train_ds,
            test_ds=test_ds,
            runs=1,
            substeps=2,
        )

        assert isinstance(results, SufficiencyOutput)

    def test_mock_run_with_kwargs(self) -> None:
        suff = Sufficiency()

        suff._train = MagicMock()
        suff._eval = MagicMock()
        suff._eval.return_value = {"test": 1.0}

        patch("torch.utils.data.DataLoader").start()

        model = MagicMock()
        train_ds = MagicMock()
        train_ds.__len__.return_value = 2
        test_ds = MagicMock()

        results = suff.run(
            model,
            train_ds=train_ds,
            test_ds=test_ds,
            runs=1,
            substeps=2,
            train_kwargs={"train": 1},
            eval_kwargs={"eval": 1},
        )

        assert suff._train.call_count == 2
        assert {"train": 1} in suff._train.call_args[0]

        assert suff._eval.call_count == 2
        assert {"eval": 1} in suff._eval.call_args[0]

        assert isinstance(results, SufficiencyOutput)

    def test_dataset_no_len(self):
        suff = Sufficiency()
        nolen_ds = MagicMock()
        delattr(nolen_ds, "__len__")
        with pytest.raises(TypeError):
            suff.run(
                model=MagicMock(),
                train_ds=nolen_ds,
                test_ds=nolen_ds,
                runs=1,
                substeps=1,
            )

    def test_dataset_len_zero(self):
        suff = Sufficiency()
        empty_ds = MagicMock()
        empty_ds.__len__.return_value = 0
        with pytest.raises(ValueError):
            suff.run(
                model=MagicMock(),
                train_ds=empty_ds,
                test_ds=empty_ds,
                runs=1,
                substeps=1,
            )

    def test_set_func_is_non_callable(self):
        suff = Sufficiency()
        with pytest.raises(TypeError):
            suff._set_func(NonCallableMagicMock())

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

        def train_task_cls_kwargs(model: nn.Module, dl: DataLoader, num: int) -> None:
            assert isinstance(num, int)

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

        def eval_kwargs(model: nn.Module, dl: DataLoader, num: int):
            """Kwargs should match parameter input"""
            assert isinstance(num, int)
            return {"test": float(num)}

        _, test_ds = load_cls_dataset()

        # Instantiate sufficiency metric
        suff = Sufficiency()
        # Set predefined eval function
        suff.set_eval_func(eval_kwargs)
        dl = DataLoader(test_ds)

        kwargs = {"num": 100}
        result = suff._eval(mock_net, dataloader=dl, kwargs=kwargs)
        assert isinstance(result, Dict)
        assert result["test"] == kwargs["num"]

    def test_plot(self):
        """Tests that a plot is generated"""
        # Only needed for plotting test
        suff = Sufficiency()

        output = SufficiencyOutput(
            measures={"test": np.array([0.2, 0.6, 0.9])},
            steps=np.array([10, 100, 1000]),
        )

        result = suff.plot(data=output)
        assert len(result) == 1
        assert isinstance(result[0], Figure)

    def test_multiplot(self):
        """Tests that the multiple plots are generated"""
        suff = Sufficiency()

        output = SufficiencyOutput(
            measures={
                "test1": np.array([0.2, 0.6, 0.9]),
                "test2": np.array([0.2, 0.6, 0.9]),
                "test3": np.array([0.2, 0.6, 0.9]),
            },
            steps=np.array([10, 100, 1000]),
        )

        result = suff.plot(data=output)
        assert len(result) == 3
        assert isinstance(result[0], Figure)


class TestSufficiencyCls:
    def test_train(self, mock_net) -> None:
        def train_task_cls(model: nn.Module, dl: DataLoader):
            # Extract first batch in dataloader
            batch = next(iter(dl))
            assert len(batch) == 2

            img, lbl = batch
            # Each image has a label
            assert img.shape[0] == lbl.shape[0]

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
        assert isinstance(result, Dict)


# Can be combined with classification using parameterize, not sure if worth
class TestSufficiencyOD:
    def test_train(self, mock_net) -> None:
        def train_task_od(model: nn.Module, dl: DataLoader):
            # Extract first batch from dataloader
            batch = next(iter(dl))
            assert len(batch)

            img, lbls, bxs = batch

            # Each image has a set of labels and boxes
            assert img.shape[0] == lbls.shape[0] == bxs.shape[0]
            # Each box as a label
            assert lbls.shape[1] == bxs.shape[1]

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
        assert isinstance(result, Dict)
