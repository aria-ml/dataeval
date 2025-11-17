from __future__ import annotations

__all__ = []

from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn

from dataeval.outputs import SufficiencyOutput
from dataeval.protocols import Dataset, EvaluationStrategy, TrainingStrategy
from dataeval.types import set_metadata

T = TypeVar("T")


@dataclass(frozen=True)
class SufficiencyConfig(Generic[T]):
    """
    Configuration for sufficiency analysis execution.

    Attributes
    ----------
    training_strategy : TrainingStrategy
        Strategy for training models on dataset subsets. Must implement
        the `train(model, dataset, indices)` method.
    evaluation_strategy : EvaluationStrategy
        Strategy for evaluating trained models. Must implement the
        `evaluate(model, dataset)` method returning metrics.
    runs : int, default 1
        Number of independent training runs to perform. Each run trains
        a fresh model from scratch.
    substeps : int, default 5
        Number of evaluation steps per run. Used for default geometric
        schedule if no custom schedule is provided.
    unit_interval : bool, default True
        Whether metrics are constrained to [0, 1]. Set True for metrics
        like accuracy, precision, recall. Set False for unbounded metrics
        like loss or error.

    Raises
    ------
    ValueError
        If runs or substeps is not greater than 1

    Examples
    --------
    Basic configuration:

    >>> training = CustomTrainingStrategy(learning_rate=0.001, epochs=10)
    >>> evaluation = CustomEvaluationStrategy(batch_size=32)
    >>> config = SufficiencyConfig(training, evaluation, runs=3, substeps=5)

    Configuration for unbounded metrics (e.g., loss):

    >>> config = SufficiencyConfig(
    ...     training,
    ...     evaluation,
    ...     runs=5,
    ...     unit_interval=False,  # For loss metrics
    ... )

    Notes
    -----
    This class is immutable (frozen=True) to ensure configuration
    cannot be accidentally modified during analysis.

    See Also
    --------
    - :class:`.TrainingStrategy`
    - :class:`.EvaluationStrategy`
    """

    training_strategy: TrainingStrategy[T]
    evaluation_strategy: EvaluationStrategy[T]
    runs: int = 1
    substeps: int = 5
    unit_interval: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.runs <= 0:
            raise ValueError(f"runs must be positive, got {self.runs}")
        if self.substeps <= 0:
            raise ValueError(f"substeps must be positive, got {self.substeps}")


def reset_parameters(model: nn.Module) -> nn.Module:
    """
    Re-initializes each layer in the model using
    the layer's defined weight_init function
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module) -> None:
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
    config : SufficiencyConfig
        Configuration object containing training/evaluation strategies and parameters.

    Warning
    -------
    Since each run is trained sequentially, increasing the parameter `runs` can significantly increase runtime.

    Notes
    -----
    Substeps is overridden by the parameter `eval_at` in :meth:`.Sufficiency.evaluate`

    The parameter based API has been removed. Use `.SufficiencyConfig` instead.

    See Also
    --------
    :class:`.SufficiencyConfig`
    """

    def __init__(
        self,
        model: nn.Module,
        train_ds: Dataset[T],
        test_ds: Dataset[T],
        config: SufficiencyConfig[T],
    ) -> None:
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.config = config

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
    def runs(self) -> int:
        return self.config.runs

    @property
    def substeps(self) -> int:
        return self.config.substeps

    @property
    def unit_interval(self) -> bool:
        return self.config.unit_interval

    @set_metadata(state=["runs", "substeps"])
    def evaluate(self, eval_at: int | Iterable[int] | None = None) -> SufficiencyOutput:
        """
        Train and evaluate a model over multiple substeps

        This function trains a model up to each step calculated from substeps. The model is then evaluated
        at that step and trained from 0 to the next step. This repeats for all substeps. Once a model has been
        trained and evaluated at all substeps, if runs is greater than one, the model weights are reset and
        the process is repeated.

        During each evaluation, the metrics returned as a dictionary by the given evaluation function are stored
        and then averaged over when all runs are complete.

        Parameters
        ----------
        eval_at : int | Iterable[int] | None, default None
            Specify this to collect metrics over a specific set of dataset lengths.
            If `None`, evaluates at each step is calculated by
            `np.geomspace` over the length of the dataset for self.substeps

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
        Default runs and substeps

        >>> config = SufficiencyConfig(
        ...     CustomTrainingStrategy(),
        ...     CustomEvaluationStrategy(),
        ... )

        >>> suff = Sufficiency(
        ...     model=model,
        ...     train_ds=train_ds,
        ...     test_ds=test_ds,
        ...     config=config,
        ... )
        >>> suff.evaluate()
        SufficiencyOutput(steps=array([  1,   3,  10,  31, 100], dtype=uint32), measures={'test': array([[1., 1., 1., 1., 1.]])}, averaged_measures={'test': array([1., 1., 1., 1., 1.])}, n_iter=1000, unit_interval=True)

        Evaluate at a single value

        >>> suff.evaluate(eval_at=50)
        SufficiencyOutput(steps=array([50]), measures={'test': array([[1.]])}, averaged_measures={'test': array([1.])}, n_iter=1000, unit_interval=True)

        Evaluating at linear steps from 0-100 inclusive

        >>> suff.evaluate(eval_at=np.arange(0, 101, 20))
        SufficiencyOutput(steps=array([  0,  20,  40,  60,  80, 100]), measures={'test': array([[1., 1., 1., 1., 1., 1.]])}, averaged_measures={'test': array([1., 1., 1., 1., 1., 1.])}, n_iter=1000, unit_interval=True)

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
        for run in range(self.runs):
            # Create a randomized set of indices to use
            indices = np.random.randint(0, self._length, size=self._length)
            # Reset the network weights to "create" an untrained model
            model = reset_parameters(self.model)
            # Run the model with each substep of data
            for iteration, substep in enumerate(ranges):
                self.config.training_strategy.train(
                    model,
                    self.train_ds,
                    indices[: int(substep)].tolist(),
                )

                # evaluate on test data
                measure = self.config.evaluation_strategy.evaluate(model, self.test_ds)

                # Keep track of each measures values
                for name, value in measure.items():
                    # Sum result into current substep iteration to be averaged later
                    value = np.array(value).ravel()
                    if name not in measures:
                        measures[name] = np.zeros(
                            (self.runs, substeps) if len(value) == 1 else (self.runs, substeps, len(value))
                        )

                    measures[name][run, iteration] = value
        return SufficiencyOutput(ranges, measures, unit_interval=self.unit_interval)
