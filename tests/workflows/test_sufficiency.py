from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)

from dataeval.outputs._workflows import (
    SufficiencyOutput,
)
from dataeval.workflows import Sufficiency
from tests.mock.data import DataEvalDataset

np.random.seed(0)
torch.manual_seed(0)


def load_cls_dataset() -> tuple[DataEvalDataset, DataEvalDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))

    train_ds = DataEvalDataset(images, labels)
    test_ds = DataEvalDataset(images, labels)

    return train_ds, test_ds


def load_od_dataset() -> tuple[DataEvalDataset, DataEvalDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))
    boxes = np.ones(shape=(1, 1, 4))

    train_ds = DataEvalDataset(images, labels, boxes)
    test_ds = DataEvalDataset(images, labels, boxes)

    return train_ds, test_ds


def eval_100(model: nn.Module, dl: DataLoader) -> dict[str, float]:
    """Eval should always return a float, and error if not"""
    return {"eval": 1.0}


def mock_ds(length: int | None):
    ds = MagicMock()
    if length is None:
        delattr(ds, "__len__")
    else:
        ds.__len__.return_value = length
    return ds


@pytest.mark.required
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
        assert isinstance(results, SufficiencyOutput)

    def test_mock_run_at_value(self) -> None:
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

        results = suff.evaluate(np.array([1]))
        assert isinstance(results, SufficiencyOutput)

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

        assert isinstance(results, SufficiencyOutput)

    def test_run_with_invalid_eval_at(self) -> None:
        suff = Sufficiency(
            model=MagicMock(),
            train_ds=mock_ds(2),
            test_ds=mock_ds(2),
            train_fn=MagicMock(),
            eval_fn=MagicMock(),
            runs=1,
            substeps=2,
        )

        with pytest.raises(ValueError):
            suff.evaluate("hello world")  # type: ignore

    def test_run_multiple_metrics(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"Accuracy": 1.0, "Precision": 1.0}
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

        output = suff.evaluate()
        assert len(output.params) == 2
        assert len(output.measures) == 2

    def test_run_classwise(self) -> None:
        eval_fn = MagicMock()
        eval_fn.return_value = {"Accuracy": np.array([0.2, 0.4, 0.6, 0.8])}
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

        output = suff.evaluate()
        assert output.params["Accuracy"].shape == (4, 3)
        assert len(output.measures) == 1

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
