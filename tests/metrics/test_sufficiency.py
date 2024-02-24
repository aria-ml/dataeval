from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from daml._internal.metrics.sufficiency import STEPS_KEY
from daml.metrics.sufficiency import Sufficiency
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


def mock_ds(len: Optional[int]):
    ds = MagicMock()
    if len is None:
        delattr(ds, "__len__")
    else:
        ds.__len__.return_value = len
    return ds


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

    def test_no_steps_key(self):
        output = {"test1": np.array([0.2, 0.6, 0.9])}
        with pytest.raises(KeyError):
            Sufficiency.project(output, "test1", 10000)

    def test_measure_length_invalid(self):
        output = {
            STEPS_KEY: np.array([10, 100]),
            "test1": np.array([0.2, 0.6, 0.9]),
        }
        with pytest.raises(ValueError):
            Sufficiency.project(output, "test1", 10000)

    @pytest.mark.parametrize("steps", [100, [100], np.array([100])])
    def test_project(self, steps):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([0.2, 0.6, 0.9]),
        }
        result = Sufficiency.project(output, "test1", steps)
        npt.assert_almost_equal(result, [0.6], decimal=4)

    def test_project_invalid_steps(self):
        output = {
            STEPS_KEY: np.array([10, 100, 1000]),
            "test1": np.array([0.2, 0.6, 0.9]),
        }
        with pytest.raises(ValueError):
            Sufficiency.project(output, "test1", 1.0)  # type: ignore
