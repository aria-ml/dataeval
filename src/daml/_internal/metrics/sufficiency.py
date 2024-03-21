from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from scipy.optimize import basinhopping
from torch.utils.data import Dataset

from daml._internal.metrics.base import EvaluateMixin

STEPS_KEY = "_STEPS_"


def f_out(n_i: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculates the line of best fit based on its free parameters

    Parameters
    ----------
    n_i : np.ndarray
        Array of sample sizes
    x : np.ndarray
        Array of inverse power curve coefficients

    Returns
    -------
    np.ndarray
        Data points for the line of best fit
    """
    return x[0] * n_i ** (-x[1]) + x[2]


def calc_params(p_i: np.ndarray, n_i: np.ndarray) -> np.ndarray:
    """
    Retrieves the inverse power curve coefficients for the line of best fit.
    Global minimization is done via basin hopping. More info on this algorithm
    can be found here: https://arxiv.org/abs/cond-mat/9803344 .

    Parameters
    ----------
    p_i : np.ndarray
        Array of corresponding losses
    n_i : np.ndarray
        Array of sample sizes

    Returns
    -------
    np.ndarray
        Array of parameters to recreate line of best fit
    """

    def f(x):
        inner = np.sum(np.square(p_i - x[0] * n_i ** (-x[1]) - x[2]))
        return inner

    res = basinhopping(f, np.array([0.5, 0.5, 0.1]))
    return res.x


def reset_parameters(model: nn.Module):
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


def validate_dataset_len(dataset: Dataset) -> int:
    if not hasattr(dataset, "__len__"):
        raise TypeError("Must provide a dataset with a length attribute")
    length: int = dataset.__len__()  # type: ignore
    if length <= 0:
        raise ValueError("Dataset length must be greater than 0")
    return length


def validate_output(data: Dict[str, np.ndarray]):
    """Ensure the sufficiency data used is not malformed"""
    if STEPS_KEY not in data:
        raise KeyError(f"{STEPS_KEY} is a required key for Sufficiency output.")
    c = len(data[STEPS_KEY])
    for m, v in data.items():
        if m == STEPS_KEY:
            continue
        if c != len(v):
            raise ValueError(
                "f{m} does not contain the expected number ({c}) of data points."
            )


def project_steps(
    measure: np.ndarray,
    steps: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Projects the measures for each value of X

    Parameters
    ----------
    measure : np.ndarray
        Measures from which to extrapolate projection
    steps : np.ndarray
        Steps of the taken measures
    projection : np.ndarray
        Steps to extrapolate
    """
    params = calc_params(p_i=(1 - measure), n_i=steps)
    return 1 - f_out(projection, params)


def plot_measure(
    name: str,
    steps: np.ndarray,
    measure: np.ndarray,
    projection: np.ndarray,
) -> Figure:
    fig = plt.figure()
    fig = cast(Figure, fig)
    fig.tight_layout()

    ax = fig.add_subplot(111)

    ax.set_title(f"{name} Sufficiency")
    ax.set_ylabel(f"{name}")
    ax.set_xlabel("Steps")

    # Plot measure over each step
    ax.scatter(steps, measure, label=f"Model Results ({name})", s=15, c="black")

    # Plot extrapolation
    ax.plot(
        projection,
        project_steps(measure, steps, projection),
        linestyle="dashed",
        label=f"Potential Model Results ({name})",
    )

    ax.legend()
    return fig


class Sufficiency(EvaluateMixin):
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
    runs : int
        Number of models to run over all subsets
    substeps : int
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
        eval_fn: Callable[
            [nn.Module, Dataset], Union[Dict[str, float], Dict[str, np.ndarray]]
        ],
        runs: int = 1,
        substeps: int = 5,
        train_kwargs: Optional[Dict[str, Any]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
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
    def train_ds(self):
        return self._train_ds

    @train_ds.setter
    def train_ds(self, value: Dataset):
        self._train_ds = value
        self._length = validate_dataset_len(value)

    @property
    def test_ds(self):
        return self._test_ds

    @test_ds.setter
    def test_ds(self, value: Dataset):
        validate_dataset_len(value)
        self._test_ds = value

    @property
    def train_fn(self) -> Callable[[nn.Module, Dataset, Sequence[int]], None]:
        return self._train_fn

    @train_fn.setter
    def train_fn(self, value: Callable[[nn.Module, Dataset, Sequence[int]], None]):
        if not callable(value):
            raise TypeError("Must provide a callable for train_fn.")
        self._train_fn = value

    @property
    def eval_fn(
        self,
    ) -> Callable[[nn.Module, Dataset], Union[Dict[str, float], Dict[str, np.ndarray]]]:
        return self._eval_fn

    @eval_fn.setter
    def eval_fn(
        self,
        value: Callable[
            [nn.Module, Dataset], Union[Dict[str, float], Dict[str, np.ndarray]]
        ],
    ):
        if not callable(value):
            raise TypeError("Must provide a callable for eval_fn.")
        self._eval_fn = value

    @property
    def train_kwargs(self) -> Dict[str, Any]:
        return self._train_kwargs

    @train_kwargs.setter
    def train_kwargs(self, value: Optional[Dict[str, Any]]):
        self._train_kwargs = {} if value is None else value

    @property
    def eval_kwargs(self) -> Dict[str, Any]:
        return self._eval_kwargs

    @eval_kwargs.setter
    def eval_kwargs(self, value: Optional[Dict[str, Any]]):
        self._eval_kwargs = {} if value is None else value

    def evaluate(self) -> Dict[str, np.ndarray]:
        """
        Creates data indices, trains models, and returns plotting data

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing the average of each measure per substep
        """

        geomshape = (
            0.01 * self._length,
            self._length,
            self.substeps,
        )  # Start, Stop, Num steps
        ranges = np.geomspace(*geomshape).astype(np.int64)

        metric_outputs = {}

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
                    indices[:substep].tolist(),
                    **self.train_kwargs,
                )

                # evaluate on test data
                output = self.eval_fn(
                    model,
                    self.test_ds,
                    **self.eval_kwargs,
                )

                # Keep track of each measures values
                for name, value in output.items():
                    if name == STEPS_KEY:
                        raise KeyError(f"Cannot use '{STEPS_KEY}' as a metric name.")

                    if name not in metric_outputs:
                        shape = (
                            (self.substeps, len(value))
                            if isinstance(value, np.ndarray)
                            else self.substeps
                        )
                        metric_outputs[name] = np.zeros(shape)

                    # Sum result into current substep iteration to be averaged later
                    metric_outputs[name][iteration] += value

        output = {STEPS_KEY: ranges}
        # The mean for each measure must be calculated before being returned
        output.update(
            {name: value / self.runs for name, value in metric_outputs.items()}
        )

        return output

    @classmethod
    def project(
        cls,
        data: Dict[str, np.ndarray],
        projection: Union[int, Sequence[int], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Projects the measures for each value of X

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dataclass containing the average of each measure per substep
        steps : Union[int, np.ndarray]
            Step or steps to project

        Raises
        ------
        KeyError
            If STEPS_KEY or measure is not a valid key
        ValueError
            If the length of data points in the measures do not match
            If the steps are not int, Sequence[int] or an ndarray
        """
        validate_output(data)
        projection = [projection] if isinstance(projection, int) else projection
        projection = (
            np.array(projection) if isinstance(projection, Sequence) else projection
        )
        if not isinstance(projection, np.ndarray):
            raise ValueError("'steps' must be an int, Sequence[int] or ndarray")

        output = {}
        output[STEPS_KEY] = projection
        for name, measure in data.items():
            if name == STEPS_KEY:
                continue

            if len(measure.shape) > 1:
                result = []
                for i in range(measure.shape[1]):
                    projected = project_steps(
                        measure[:, i], data[STEPS_KEY], projection
                    )
                    result.append(projected)
                output[name] = np.array(result).T
            else:
                output[name] = project_steps(measure, data[STEPS_KEY], projection)
        return output

    @classmethod
    def plot(
        cls, data: Dict[str, np.ndarray], class_names: Optional[Sequence[str]] = None
    ) -> List[Figure]:
        """Plotting function for data sufficiency tasks

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dataclass containing the average of each measure per substep

        Returns
        -------
        List[plt.Figure]
            List of Figures for each measure

        Raises
        ------
        KeyError
            If STEPS_KEY or measure is not a valid key
        ValueError
            If the length of data points in the measures do not match
        """
        validate_output(data)

        # X, y data
        steps = data[STEPS_KEY]

        # Extrapolation parameters
        last_X = steps[-1]
        geomshape = (0.01 * last_X, last_X * 4, len(steps))
        extrapolated = np.geomspace(*geomshape).astype(np.int64)

        # Stores all plots
        plots = []

        # Create a plot for each measure on one figure
        for name, measure in data.items():
            if name == STEPS_KEY:
                continue

            if len(measure.shape) > 1:
                if class_names is not None and measure.shape[1] != len(class_names):
                    raise IndexError("Class name count does not align with measures")
                for i in range(measure.shape[1]):
                    class_name = str(i) if class_names is None else class_names[i]
                    fig = plot_measure(
                        name + "_" + class_name,
                        steps,
                        measure[:, i],
                        extrapolated,
                    )
                    plots.append(fig)

            else:
                fig = plot_measure(name, steps, measure, extrapolated)
                plots.append(fig)

        return plots
