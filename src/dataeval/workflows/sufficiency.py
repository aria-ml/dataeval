from __future__ import annotations

__all__ = []

from typing import Any, Callable, Generic, Iterable, Mapping, Sequence, Sized, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataeval.outputs import SufficiencyOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike


def reset_parameters(model: nn.Module) -> nn.Module:
    """
    Re-initializes each layer in the model using
    the layer's defined weight_init function
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # Check if the current module has reset_parameters
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()  # type: ignore

    # Applies fn recursively to every submodule see:
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    return model.apply(fn=weight_reset)


def validate_dataset_len(dataset: Dataset[Any]) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("Must provide a dataset with a length attribute")
    length: int = len(dataset)
    if length <= 0:
        raise ValueError("Dataset length must be greater than 0")
    return length


T = TypeVar("T")


class Sufficiency(Generic[T]):
    """
    Project dataset :term:`sufficiency<Sufficiency>` using given a model and evaluation criteria.

    Parameters
    ----------
    model : nn.Module
        Model that will be trained for each subset of data
    train_ds : torch.Dataset
        Full training data that will be split for each run
    test_ds : torch.Dataset
        Data that will be used for every run's evaluation
    train_fn : Callable[[nn.Module, Dataset, Sequence[int]], None]
        Function which takes a model (torch.nn.Module), a dataset
        (torch.utils.data.Dataset), indices to train on and executes model
        training against the data.
    eval_fn : Callable[[nn.Module, Dataset], Mapping[str, float | ArrayLike]]
        Function which takes a model (torch.nn.Module), a dataset
        (torch.utils.data.Dataset) and returns a dictionary of metric
        values (Mapping[str, float]) which is used to assess model performance
        given the model and data.
    runs : int, default 1
        Number of models to run over all subsets
    substeps : int, default 5
        Total number of dataset partitions that each model will train on
    train_kwargs : Mapping | None, default None
        Additional arguments required for custom training function
    eval_kwargs : Mapping | None, default None
        Additional arguments required for custom evaluation function
    """

    def __init__(
        self,
        model: nn.Module,
        train_ds: Dataset[T],
        test_ds: Dataset[T],
        train_fn: Callable[[nn.Module, Dataset[T], Sequence[int]], None],
        eval_fn: Callable[[nn.Module, Dataset[T]], Mapping[str, float] | Mapping[str, ArrayLike]],
        runs: int = 1,
        substeps: int = 5,
        train_kwargs: Mapping[str, Any] | None = None,
        eval_kwargs: Mapping[str, Any] | None = None,
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

    @property
    def train_ds(self) -> Dataset[T]:
        return self._train_ds

    @train_ds.setter
    def train_ds(self, value: Dataset[T]) -> None:
        self._train_ds = value
        self._length = validate_dataset_len(value)

    @property
    def test_ds(self) -> Dataset[T]:
        return self._test_ds

    @test_ds.setter
    def test_ds(self, value: Dataset[T]) -> None:
        validate_dataset_len(value)
        self._test_ds = value

    @property
    def train_fn(self) -> Callable[[nn.Module, Dataset[T], Sequence[int]], None]:
        return self._train_fn

    @train_fn.setter
    def train_fn(self, value: Callable[[nn.Module, Dataset[T], Sequence[int]], None]) -> None:
        if not callable(value):
            raise TypeError("Must provide a callable for train_fn.")
        self._train_fn = value

    @property
    def eval_fn(
        self,
    ) -> Callable[[nn.Module, Dataset[T]], Mapping[str, float] | Mapping[str, ArrayLike]]:
        return self._eval_fn

    @eval_fn.setter
    def eval_fn(
        self,
        value: Callable[[nn.Module, Dataset[T]], Mapping[str, float] | Mapping[str, ArrayLike]],
    ) -> None:
        if not callable(value):
            raise TypeError("Must provide a callable for eval_fn.")
        self._eval_fn = value

    @property
    def train_kwargs(self) -> Mapping[str, Any]:
        return self._train_kwargs

    @train_kwargs.setter
    def train_kwargs(self, value: Mapping[str, Any] | None) -> None:
        self._train_kwargs = {} if value is None else value

    @property
    def eval_kwargs(self) -> Mapping[str, Any]:
        return self._eval_kwargs

    @eval_kwargs.setter
    def eval_kwargs(self, value: Mapping[str, Any] | None) -> None:
        self._eval_kwargs = {} if value is None else value

    @set_metadata(state=["runs", "substeps"])
    def evaluate(self, eval_at: int | Iterable[int] | None = None) -> SufficiencyOutput:
        """
        Creates data indices, trains models, and returns plotting data

        Parameters
        ----------
        eval_at : int | Iterable[int] | None, default None
            Specify this to collect accuracies over a specific set of dataset lengths, rather
            than letting :term:`sufficiency<Sufficiency>` internally create the lengths to evaluate at.

        Returns
        -------
        SufficiencyOutput
            Dataclass containing the average of each measure per substep

        Raises
        ------
        ValueError
            If `eval_at` is not numerical

        Examples
        --------
        >>> suff = Sufficiency(
        ...     model=model,
        ...     train_ds=train_ds,
        ...     test_ds=test_ds,
        ...     train_fn=train_fn,
        ...     eval_fn=eval_fn,
        ...     runs=3,
        ...     substeps=5,
        ... )
        >>> suff.evaluate()
        SufficiencyOutput(steps=array([  1,   3,  10,  31, 100], dtype=uint32), measures={'test': array([1., 1., 1., 1., 1.])}, n_iter=1000)
        """  # noqa: E501
        if eval_at is not None:
            ranges = np.asarray(list(eval_at) if isinstance(eval_at, Iterable) else [eval_at])
            if not np.issubdtype(ranges.dtype, np.number):
                raise ValueError("'eval_at' must consist of numerical values")
        else:
            geomshape = (
                0.01 * self._length,
                self._length,
                self.substeps,
            )  # Start, Stop, Num steps
            ranges = np.geomspace(*geomshape, dtype=np.uint32)
        substeps = len(ranges)
        measures = {}

        # Run each model over all indices
        for _ in range(self.runs):
            # Create a randomized set of indices to use
            indices = np.random.randint(0, self._length, size=self._length)
            # Reset the network weights to "create" an untrained model
            model = reset_parameters(self.model)
            # Run the model with each substep of data
            for iteration, substep in enumerate(ranges):
                # train on subset of train data
                self.train_fn(
                    model,
                    self.train_ds,
                    indices[: int(substep)].tolist(),
                    **self.train_kwargs,
                )

                # evaluate on test data
                measure = self.eval_fn(model, self.test_ds, **self.eval_kwargs)

                # Keep track of each measures values
                for name, value in measure.items():
                    # Sum result into current substep iteration to be averaged later
                    value = np.array(value).ravel()
                    if name not in measures:
                        measures[name] = np.zeros(substeps if len(value) == 1 else (substeps, len(value)))
                    measures[name][iteration] += value

        # The mean for each measure must be calculated before being returned
        measures = {k: (v / self.runs).T for k, v in measures.items()}
        return SufficiencyOutput(ranges, measures)
