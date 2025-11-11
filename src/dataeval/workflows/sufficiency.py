from __future__ import annotations

__all__ = []

from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn

from dataeval.outputs import SufficiencyOutput
from dataeval.protocols import ArrayLike, Dataset, EvaluationStrategy, TrainingStrategy
from dataeval.types import set_metadata

T = TypeVar("T")


@dataclass(frozen=True)
class SufficiencyConfig(Generic[T]):
    """
    Configuration for sufficiency analysis execution.

    Parameters
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
    config : SufficiencyConfig or None
        Configuration object containing training/evaluation strategies and parameters.
        If not provided, must use legacy parameters (train_fn, eval_fn, etc.)
    train_fn : Callable[[nn.Module, Dataset, Sequence[int]], None]
        DEPRECATED Function which takes a model, a dataset, and indices to train on and then executes model
        training against the data.
    eval_fn : Callable[[nn.Module, Dataset], Mapping[str, float | ArrayLike]]
        DEPRECATED Function which takes a model, a dataset and returns a dictionary of metric
        values which is used to assess model performance
        given the model and data.
    runs : int, default 1
        DEPRECATED Number of models to train over the entire dataset.
    substeps : int, default 5
        DEPRECATED The number of steps that each model will be trained and evaluated on.
    train_kwargs : Mapping | None, default None
        DEPRECATED Additional arguments required for custom training function
    eval_kwargs : Mapping | None, default None
        DEPRECATED Additional arguments required for custom evaluation function
    unit_interval : bool, default True
        DEPRECATED Constrains the power law to the interval [0, 1].
        Set True (default) for metrics such as accuracy, precision,
        and recall which are defined to take values on [0,1]. Set False for metrics not on the unit interval.

    Warning
    -------
    Since each run is trained sequentially, increasing the parameter `runs` can significantly increase runtime.

    Notes
    -----
    Substeps is overridden by the parameter `eval_at` in :meth:`.Sufficiency.evaluate`

    The constructor supports two signatures during transition:

    NEW (recommended):
        Sufficiency(model, train_ds, test_ds, config)

    OLD (deprecated):
        Sufficiency(model, train_ds, test_ds, train_fn, eval_fn, ...)

    The old signature will be removed in a future version.
    """

    def __init__(
        self,
        model: nn.Module,
        train_ds: Dataset[T],
        test_ds: Dataset[T],
        config: SufficiencyConfig[T] | None = None,
        train_fn: Callable[[nn.Module, Dataset[T], Sequence[int]], None] | None = None,
        eval_fn: Callable[[nn.Module, Dataset[T]], Mapping[str, float] | Mapping[str, ArrayLike]] | None = None,
        runs: int = 1,
        substeps: int = 5,
        train_kwargs: Mapping[str, Any] | None = None,
        eval_kwargs: Mapping[str, Any] | None = None,
        unit_interval: bool = True,
    ) -> None:
        # Detect which signature is being used
        using_new_api: bool = config is not None
        using_old_api = train_fn is not None or eval_fn is not None

        # Validate signature usage
        if using_new_api and using_old_api:
            raise ValueError(
                "Cannot provide both config and legacy parameters (train_fn/eval_fn). "
                "Use either the new API with SufficiencyConfig or the old API, not both."
            )

        if not using_new_api and not using_old_api:
            raise ValueError(
                "Must provide either config (new API) or train_fn/eval_fn (old API). "
                "Recommended: use SufficiencyConfig for new code."
            )

        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds

        if using_new_api:
            self.config: SufficiencyConfig[T] = config
        # Adapts old API parameters into new config API
        else:
            training_strategy = _FunctionTrainingStrategy[T](train_fn, train_kwargs)  # pyright: ignore[reportArgumentType]
            evaluation_strategy = _FunctionEvaluationStrategy[T](eval_fn, eval_kwargs)  # pyright: ignore[reportArgumentType]

            self.config: SufficiencyConfig[T] = SufficiencyConfig(
                training_strategy=training_strategy,
                evaluation_strategy=evaluation_strategy,
                runs=runs,
                substeps=substeps,
                unit_interval=unit_interval,
            )

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
        """
        Access training function

        DEPRECATED: Use config.training_strategy instead

        Returns the underlying training function if using old API, or raises AttributeError
        if using the new API with strategy objects
        """
        # Unwrap the original function if using adapter
        if isinstance(self.config.training_strategy, _FunctionTrainingStrategy):
            return self.config.training_strategy._train_fn

        raise AttributeError(
            "train_fn property only available when using legacy API. Use config.training_strategy instead."
        )

    @property
    def eval_fn(
        self,
    ) -> Callable[[nn.Module, Dataset[T]], Mapping[str, float] | Mapping[str, ArrayLike]]:
        """
        Access evaluation function

        DEPRECATED: Use config.evaluation_strategy instead

        Returns the underlying evaluation function if using old API, or raises AttributeError
        if using the new API with strategy objects
        """
        # Unwrap the original function if using adapter
        if isinstance(self.config.evaluation_strategy, _FunctionEvaluationStrategy):
            return self.config.evaluation_strategy._eval_fn

        raise AttributeError(
            "eval_fn property only available when using legacy API. Use config.evaluation_strategy instead."
        )

    @property
    def train_kwargs(self) -> Mapping[str, Any]:
        """
        Access training kwargs.

        DEPRECATED: Specify parameters in your TrainingStrategy implementation.

        Returns kwargs if using old API, empty dict if using new API.
        """
        if isinstance(self.config.training_strategy, _FunctionTrainingStrategy):
            return self.config.training_strategy._kwargs
        return {}

    @property
    def eval_kwargs(self) -> Mapping[str, Any]:
        """
        Access evaluation kwargs.

        DEPRECATED: Specify parameters in your EvaluationStrategy implementation.

        Returns kwargs if using old API, empty dict if using new API.
        """
        if isinstance(self.config.evaluation_strategy, _FunctionEvaluationStrategy):
            return self.config.evaluation_strategy._kwargs
        return {}

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
        SufficiencyOutput(steps=array([  1,   3,  10,  31, 100], dtype=uint32), measures={'test': array([[1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]])}, averaged_measures={'test': array([1., 1., 1., 1., 1.])}, n_iter=1000, unit_interval=True)

        Evaluate at a single value

        >>> suff = Sufficiency(
        ...     model=model,
        ...     train_ds=train_ds,
        ...     test_ds=test_ds,
        ...     train_fn=train_fn,
        ...     eval_fn=eval_fn,
        ... )
        >>> suff.evaluate(eval_at=50)
        SufficiencyOutput(steps=array([50]), measures={'test': array([[1.]])}, averaged_measures={'test': array([1.])}, n_iter=1000, unit_interval=True)

        Evaluating at linear steps from 0-100 inclusive

        >>> suff = Sufficiency(
        ...     model=model,
        ...     train_ds=train_ds,
        ...     test_ds=test_ds,
        ...     train_fn=train_fn,
        ...     eval_fn=eval_fn,
        ... )
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


class _FunctionTrainingStrategy(Generic[T]):
    """
    Internal adapter that wraps legacy training functions.

    Allows old train_fn + train_kwargs to work with new Strategy interface.
    """

    def __init__(
        self,
        train_fn: Callable[[nn.Module, Dataset[T], Sequence[int]], None],
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._train_fn = train_fn
        self._kwargs = kwargs if kwargs is not None else {}

    def train(self, model: nn.Module, dataset: Dataset[T], indices: Sequence[int]) -> None:
        """Delegate to wrapped function with kwargs."""
        self._train_fn(model, dataset, indices, **self._kwargs)


class _FunctionEvaluationStrategy(Generic[T]):
    """
    Internal adapter that wraps legacy evaluation functions.

    Allows old eval_fn + eval_kwargs to work with new Strategy interface.
    """

    def __init__(
        self,
        eval_fn: Callable[[nn.Module, Dataset[T]], Mapping[str, float | ArrayLike]],
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._eval_fn = eval_fn
        self._kwargs = kwargs if kwargs is not None else {}

    def evaluate(self, model: nn.Module, dataset: Dataset[T]) -> Mapping[str, float | ArrayLike]:
        """Delegate to wrapped function with kwargs."""
        return self._eval_fn(model, dataset, **self._kwargs)
