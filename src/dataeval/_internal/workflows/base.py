from abc import abstractmethod
from typing import Any, Callable, Sequence

import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset


class Sufficiency:
    """
    Project dataset sufficiency using given a model and evaluation criteria

    Parameters
    ----------
    model : nn.Module
        Model that will be trained for each subset of data
    train_ds : Dataset
        Full training data that will be split for each run
    test_ds : Dataset
        Data that will be used for every run's evaluation
    train_fn : Callable[[nn.Module, Dataset, Sequence[int]], None]
        Function which takes a model (torch.nn.Module), a dataset
        (torch.utils.data.Dataset), indices to train on and executes model
        training against the data.
    eval_fn : Callable[[nn.Module, Dataset], Dict[str, float]]
        Function which takes a model (torch.nn.Module), a dataset
        (torch.utils.data.Dataset) and returns a dictionary of metric
        values (Dict[str, float]) which is used to assess model performance
        given the model and data.
    runs : int, default 1
        Number of models to run over all subsets
    substeps : int, default 5
        Total number of dataset partitions that each model will train on
    train_kwargs : Dict[str, Any] | None, default None
        Additional arguments required for custom training function
    eval_kwargs : Dict[str, Any] | None, default None
        Additional arguments required for custom evaluation function
    """

    def __init__(
        self,
        model: nn.Module,
        train_ds: Dataset,
        test_ds: Dataset,
        train_fn: Callable[[nn.Module, Dataset, Sequence[int]], None],
        eval_fn: Callable[[nn.Module, Dataset], dict[str, float] | dict[str, NDArray]],
        runs: int = 1,
        substeps: int = 5,
        train_kwargs: dict[str, Any] | None = None,
        eval_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.runs = runs
        self.substeps = substeps
        self.train_kwargs = train_kwargs
        self.eval_kwargs = eval_kwargs

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> dict[str, NDArray]:
        raise NotImplementedError
