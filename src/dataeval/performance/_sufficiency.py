__all__ = []

from collections.abc import Iterable, Sized
from typing import Any, Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn
from pydantic import field_validator

from dataeval.performance._aggregator import ResultAggregator
from dataeval.performance._output import SufficiencyOutput
from dataeval.performance.schedules import GeometricSchedule, ManualSchedule
from dataeval.protocols import Dataset, EvaluationSchedule, EvaluationStrategy, TrainingStrategy
from dataeval.types import Evaluator, EvaluatorConfig, set_metadata

DEFAULT_SUFFICIENCY_RUNS = 1
DEFAULT_SUFFICIENCY_SUBSTEPS = 5
DEFAULT_SUFFICIENCY_UNIT_INTERVAL = True

T = TypeVar("T")
S = TypeVar("S")


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


class Sufficiency(Evaluator, Generic[T]):
    """
    Analyze how much training data is needed for target model performance.

    Trains models on progressively larger data subsets, evaluates at each step,
    and fits power law curves to predict performance on larger datasets.

    Parameters
    ----------
    model : nn.Module
        Model to train (reset for each run)
    train_ds : torch.Dataset
        Full training data
    test_ds : torch.Dataset
        Test/validation data
    training_strategy : TrainingStrategy or None, default None
        Strategy for training models. If None, uses config.training_strategy.
    evaluation_strategy : EvaluationStrategy or None, default None
        Strategy for evaluating models. If None, uses config.evaluation_strategy.
    runs : int or None, default None
        Number of independent training runs. If None, uses config.runs (default 1).
    substeps : int or None, default None
        Number of evaluation steps per run. If None, uses config.substeps (default 5).
    unit_interval : bool or None, default None
        Whether metrics are constrained to [0, 1]. If None, uses config.unit_interval (default True).
    config : Sufficiency.Config or None, default None
        Optional configuration object. Parameters passed directly to __init__ override config values.

    Warning
    -------
    Since each run is trained sequentially, increasing the parameter `runs` can significantly increase runtime.

    Notes
    -----
    Datasets are immutable after construction. To use different data, create a new instance.

    Multiple runs average results to reduce variance.

    Parameters passed directly to __init__ override config defaults.

    See Also
    --------
    :class:`.Sufficiency.Config` : Configuration object
    :class:`.SufficiencyOutput` : Results with measures and projections
    """

    class Config(EvaluatorConfig, Generic[S]):
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
        >>> config = Sufficiency.Config(
        ...     training_strategy=training,
        ...     evaluation_strategy=evaluation,
        ...     runs=3,
        ...     substeps=5,
        ... )

        Configuration for unbounded metrics (e.g., loss):

        >>> config = Sufficiency.Config(
        ...     training_strategy=training,
        ...     evaluation_strategy=evaluation,
        ...     runs=5,
        ...     unit_interval=False,  # For loss metrics
        ... )

        See Also
        --------
        - :class:`.TrainingStrategy`
        - :class:`.EvaluationStrategy`
        """

        training_strategy: TrainingStrategy[S] | None = None
        evaluation_strategy: EvaluationStrategy[S] | None = None
        runs: int = DEFAULT_SUFFICIENCY_RUNS
        substeps: int = DEFAULT_SUFFICIENCY_SUBSTEPS
        unit_interval: bool = DEFAULT_SUFFICIENCY_UNIT_INTERVAL

        @field_validator("runs", "substeps")
        @classmethod
        def validate_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("must be positive")
            return v

    runs: int
    substeps: int
    unit_interval: bool
    training_strategy: TrainingStrategy[T]
    evaluation_strategy: EvaluationStrategy[T]
    config: Config[T]

    def __init__(
        self,
        model: nn.Module,
        train_ds: Dataset[T],
        test_ds: Dataset[T],
        training_strategy: TrainingStrategy[T] | None = None,
        evaluation_strategy: EvaluationStrategy[T] | None = None,
        runs: int | None = None,
        substeps: int | None = None,
        unit_interval: bool | None = None,
        config: Config[T] | None = None,
    ) -> None:
        self.model = model

        # Validate and store datasets
        self._length = validate_dataset_len(train_ds)
        self._train_ds = train_ds

        validate_dataset_len(test_ds)
        self._test_ds = test_ds

        super().__init__(locals(), exclude={"model", "train_ds", "test_ds"})

        # Validate parameters
        if self.runs <= 0:
            raise ValueError(f"runs must be positive, got {self.runs}")
        if self.substeps <= 0:
            raise ValueError(f"substeps must be positive, got {self.substeps}")

    @property
    def train_ds(self) -> Dataset[T]:
        """
        Training dataset (read-only)

        Notes
        -----
        This property is read-only. To use a different training dataset, create a new Sufficiency instance
        """
        return self._train_ds

    @property
    def test_ds(self) -> Dataset[T]:
        """
        Test dataset (read-only)

        Notes
        -----
        This property is read-only. To use a different test dataset, create a new Sufficiency instance
        """
        return self._test_ds

    def _create_schedule(self, schedule: EvaluationSchedule | int | Iterable[int] | None) -> EvaluationSchedule:
        """
        Convert schedule parameter into an EvaluationSchedule object.

        Handles auto-wrapping of int and iterable inputs for convenience.

        Parameters
        ----------
        schedule : EvaluationSchedule or int or Iterable[int] or None, default None
            Custom schedule object, specific evaluation points, or None for default geometric schedule

        Returns
        -------
        EvaluationSchedule
            Concrete schedule object
        """
        if schedule is None:
            return GeometricSchedule(substeps=self.substeps)
        if isinstance(schedule, EvaluationSchedule):
            return schedule
        # Wrap int or iterable
        return ManualSchedule(schedule)

    def _execute_run(self, run_index: int, steps: Iterable[int], aggregator: ResultAggregator) -> None:
        """
        Execute a single training run across all evaluation steps.

        This method makes temporal coupling explicit: model must be reset
        before training, and training must occur before evaluation at each step.

        Parameters
        ----------
        run_index : int
            Index of current run (0-based)
        steps : NDArray[uint32]
            Evaluation points (dataset sizes)
        aggregator : ResultAggregator
            Accumulator for storing results
        """
        # Step 1: Create randomized indices for this run
        indices = np.random.randint(0, self._length, size=self._length)

        # Step 2: Reset model to fresh initialization (temporal coupling explicit)
        model = reset_parameters(self.model)

        # Step 3: Train and evaluate at each step (temporal coupling explicit)
        for step_index, dataset_size in enumerate(steps):
            # Train on subset
            self.training_strategy.train(model, self.train_ds, indices[: int(dataset_size)].tolist())

            # Evaluate on test set
            metrics = self.evaluation_strategy.evaluate(model, self.test_ds)

            # Store results
            for metric_name, metric_value in metrics.items():
                aggregator.add_result(run=run_index, step=step_index, metric_name=metric_name, value=metric_value)

    @set_metadata(state=["runs", "substeps"])
    def evaluate(self, schedule: EvaluationSchedule | int | Iterable[int] | None = None) -> SufficiencyOutput:
        """
        Train and evaluate model across multiple dataset sizes.

        This function trains a model up to each step calculated from substeps. The model is then evaluated
        at that step and trained from 0 to the next step. This repeats for all substeps. Once a model has been
        trained and evaluated at all substeps, if runs is greater than one, the model weights are reset and
        the process is repeated.

        During each evaluation, the metrics returned as a dictionary by the given evaluation function are stored
        and then averaged over when all runs are complete.

        Parameters
        ----------
        schedule : EvaluationStrategy or int or Iterable[int] or None, default None
            Specify this to collect metrics over a specific set of dataset lengths.
            If `None`, evaluates at each step calculated by
            `np.geomspace` over the length of the dataset

        Returns
        -------
        SufficiencyOutput
            Contains steps, measures, averaged_measures, and params

        Examples
        --------
        >>> sufficiency = Sufficiency(
        ...     model=model,
        ...     train_ds=train_ds,
        ...     test_ds=test_ds,
        ...     training_strategy=CustomTrainingStrategy(),
        ...     evaluation_strategy=CustomEvaluationStrategy(),
        ... )

        Default runs and substeps:

        >>> output = sufficiency.evaluate()

        Evaluate at specific points:

        >>> output = sufficiency.evaluate(schedule=[100, 500, 1000])

        Evaluate at a custom geometric spacing

        >>> from dataeval.performance.schedules import GeometricSchedule
        >>> output = sufficiency.evaluate(schedule=GeometricSchedule(substeps=20))

        Evaluate at custom linear steps from 0-100 inclusive

        >>> class LinearSchedule:
        ...     def get_steps(self, dataset_length):
        ...         return np.arange(0, 101, 20)
        >>> output = sufficiency.evaluate(schedule=LinearSchedule())
        """

        # Create evaluation schedule
        schedule_obj = self._create_schedule(schedule=schedule)
        steps = schedule_obj.get_steps(self._length)

        aggregator = ResultAggregator(runs=self.runs, substeps=len(steps))

        # Execute all runs
        for run_index in range(self.runs):
            self._execute_run(run_index=run_index, steps=steps, aggregator=aggregator)

        # Create output
        results = aggregator.get_results()
        return SufficiencyOutput(steps=steps, measures=results, unit_interval=self.unit_interval)
