from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import basinhopping
from torch.utils.data import Dataset

from dataeval._internal.interop import as_numpy
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class SufficiencyOutput(OutputMetadata):
    """
    Output class for :class:`Sufficiency` workflow

    Attributes
    ----------
    steps : NDArray
        Array of sample sizes
    params : Dict[str, NDArray]
        Inverse power curve coefficients for the line of best fit for each measure
    measures : Dict[str, NDArray]
        Average of values observed for each sample size step for each measure
    """

    steps: NDArray[np.uint32]
    params: dict[str, NDArray[np.float64]]
    measures: dict[str, NDArray[np.float64]]

    def __post_init__(self):
        c = len(self.steps)
        if set(self.params) != set(self.measures):
            raise ValueError("params and measures have a key mismatch")
        for m, v in self.measures.items():
            c_v = v.shape[1] if v.ndim > 1 else len(v)
            if c != c_v:
                raise ValueError(f"{m} does not contain the expected number ({c}) of data points.")

    @set_metadata("dataeval.workflows.SufficiencyOutput")
    def project(
        self,
        projection: int | Iterable[int],
    ) -> SufficiencyOutput:
        """Projects the measures for each value of X

        Parameters
        ----------
        projection : int | Iterable[int]
            Step or steps to project

        Returns
        -------
        SufficiencyOutput
            Dataclass containing the projected measures per projection

        Raises
        ------
        ValueError
            If the length of data points in the measures do not match
            If `projection` is not numerical
        """
        projection = np.asarray(list(projection) if isinstance(projection, Iterable) else [projection])

        if not np.issubdtype(projection.dtype, np.number):
            raise ValueError("'projection' must consist of numerical values")

        output = {}
        for name, measures in self.measures.items():
            if measures.ndim > 1:
                result = []
                for i in range(len(measures)):
                    projected = project_steps(self.params[name][i], projection)
                    result.append(projected)
                output[name] = np.array(result)
            else:
                output[name] = project_steps(self.params[name], projection)
        return SufficiencyOutput(projection, self.params, output)

    def plot(self, class_names: Sequence[str] | None = None) -> list[Figure]:
        """Plotting function for data sufficiency tasks

        Parameters
        ----------
        class_names : Sequence[str] | None, default None
            List of class names

        Returns
        -------
        list[plt.Figure]
            List of Figures for each measure

        Raises
        ------
        ValueError
            If the length of data points in the measures do not match
        """
        # Extrapolation parameters
        last_X = self.steps[-1]
        geomshape = (0.01 * last_X, last_X * 4, len(self.steps))
        extrapolated = np.geomspace(*geomshape).astype(np.int64)

        # Stores all plots
        plots = []

        # Create a plot for each measure on one figure
        for name, measures in self.measures.items():
            if measures.ndim > 1:
                if class_names is not None and len(measures) != len(class_names):
                    raise IndexError("Class name count does not align with measures")
                for i, measure in enumerate(measures):
                    class_name = str(i) if class_names is None else class_names[i]
                    fig = plot_measure(
                        f"{name}_{class_name}",
                        self.steps,
                        measure,
                        self.params[name][i],
                        extrapolated,
                    )
                    plots.append(fig)

            else:
                fig = plot_measure(name, self.steps, measures, self.params[name], extrapolated)
                plots.append(fig)

        return plots

    def inv_project(self, targets: Mapping[str, ArrayLike]) -> dict[str, NDArray[np.float64]]:
        """
        Calculate training samples needed to achieve target model metric values.

        Parameters
        ----------
        targets : Mapping[str, ArrayLike]
            Mapping of target metric scores (from 0.0 to 1.0) that we want
            to achieve, where the key is the name of the metric.

        Returns
        -------
        dict[str, NDArray]
            List of the number of training samples needed to achieve each
            corresponding entry in targets
        """

        projection = {}

        for name, target in targets.items():
            tarray = as_numpy(target)
            if name not in self.measures:
                continue

            measure = self.measures[name]
            if measure.ndim > 1:
                projection[name] = np.zeros((len(measure), len(tarray)))
                for i in range(len(measure)):
                    projection[name][i] = inv_project_steps(
                        self.params[name][i], tarray[i] if tarray.ndim == measure.ndim else tarray
                    )
            else:
                projection[name] = inv_project_steps(self.params[name], tarray)

        return projection


def f_out(n_i: NDArray, x: NDArray) -> NDArray:
    """
    Calculates the line of best fit based on its free parameters

    Parameters
    ----------
    n_i : NDArray
        Array of sample sizes
    x : NDArray
        Array of inverse power curve coefficients

    Returns
    -------
    NDArray
        Data points for the line of best fit
    """
    return x[0] * n_i ** (-x[1]) + x[2]


def f_inv_out(y_i: NDArray, x: NDArray) -> NDArray[np.uint64]:
    """
    Inverse function for f_out()

    Parameters
    ----------
    y_i : NDArray
        Data points for the line of best fit
    x : NDArray
        Array of inverse power curve coefficients

    Returns
    -------
    NDArray
        Array of sample sizes
    """
    n_i = ((y_i - x[2]) / x[0]) ** (-1 / x[1])
    return np.asarray(n_i, dtype=np.uint64)


def calc_params(p_i: NDArray, n_i: NDArray, niter: int) -> NDArray:
    """
    Retrieves the inverse power curve coefficients for the line of best fit.
    Global minimization is done via basin hopping. More info on this algorithm
    can be found here: https://arxiv.org/abs/cond-mat/9803344 .

    Parameters
    ----------
    p_i : NDArray
        Array of corresponding losses
    n_i : NDArray
        Array of sample sizes
    niter : int
        Number of iterations to perform in the basin-hopping
        numerical process to curve-fit p_i

    Returns
    -------
    NDArray
        Array of parameters to recreate line of best fit
    """

    def is_valid(f_new, x_new, f_old, x_old):
        return f_new != np.nan

    def f(x):
        try:
            return np.sum(np.square(p_i - f_out(n_i, x)))
        except RuntimeWarning:
            return np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        res = basinhopping(
            f,
            np.array([0.5, 0.5, 0.1]),
            niter=niter,
            stepsize=1.0,
            minimizer_kwargs={"method": "Powell"},
            accept_test=is_valid,
            niter_success=200,
        )
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


def project_steps(params: NDArray, projection: NDArray) -> NDArray:
    """Projects the measures for each value of X

    Parameters
    ----------
    params : NDArray
        Inverse power curve coefficients used to calculate projection
    projection : NDArray
        Steps to extrapolate

    Returns
    -------
    NDArray
        Extrapolated measure values at each projection step

    """
    return 1 - f_out(projection, params)


def inv_project_steps(params: NDArray, targets: NDArray) -> NDArray[np.uint64]:
    """Inverse function for project_steps()

    Parameters
    ----------
    params : NDArray
        Inverse power curve coefficients used to calculate projection
    targets : NDArray
        Desired measure values

    Returns
    -------
    NDArray
        Array of sample sizes, or 0 if overflow
    """
    steps = f_inv_out(1 - np.array(targets), params)
    steps[np.isnan(steps)] = 0
    return np.ceil(steps)


def get_curve_params(measures: dict[str, NDArray], ranges: NDArray, niter: int) -> dict[str, NDArray]:
    """Calculates and aggregates parameters for both single and multi-class metrics"""
    output = {}
    for name, measure in measures.items():
        measure = cast(np.ndarray, measure)
        if measure.ndim > 1:
            result = []
            for value in measure:
                result.append(calc_params(1 - value, ranges, niter))
            output[name] = np.array(result)
        else:
            output[name] = calc_params(1 - measure, ranges, niter)
    return output


def plot_measure(
    name: str,
    steps: NDArray,
    measure: NDArray,
    params: NDArray,
    projection: NDArray,
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
        project_steps(params, projection),
        linestyle="dashed",
        label=f"Potential Model Results ({name})",
    )

    ax.legend()
    return fig


class Sufficiency:
    """
    Project dataset sufficiency using given a model and evaluation criteria

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
        train_ds: Dataset,
        test_ds: Dataset,
        train_fn: Callable[[nn.Module, Dataset, Sequence[int]], None],
        eval_fn: Callable[[nn.Module, Dataset], Mapping[str, float] | Mapping[str, ArrayLike]],
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
    ) -> Callable[[nn.Module, Dataset], dict[str, float] | Mapping[str, ArrayLike]]:
        return self._eval_fn

    @eval_fn.setter
    def eval_fn(
        self,
        value: Callable[[nn.Module, Dataset], dict[str, float] | Mapping[str, ArrayLike]],
    ):
        if not callable(value):
            raise TypeError("Must provide a callable for eval_fn.")
        self._eval_fn = value

    @property
    def train_kwargs(self) -> Mapping[str, Any]:
        return self._train_kwargs

    @train_kwargs.setter
    def train_kwargs(self, value: Mapping[str, Any] | None):
        self._train_kwargs = {} if value is None else value

    @property
    def eval_kwargs(self) -> Mapping[str, Any]:
        return self._eval_kwargs

    @eval_kwargs.setter
    def eval_kwargs(self, value: Mapping[str, Any] | None):
        self._eval_kwargs = {} if value is None else value

    @set_metadata("dataeval.workflows", ["runs", "substeps"])
    def evaluate(self, eval_at: int | Iterable[int] | None = None, niter: int = 1000) -> SufficiencyOutput:
        """
        Creates data indices, trains models, and returns plotting data

        Parameters
        ----------
        eval_at : int | Iterable[int] | None, default None
            Specify this to collect accuracies over a specific set of dataset lengths, rather
            than letting Sufficiency internally create the lengths to evaluate at.
        niter : int, default 1000
            Iterations to perform when using the basin-hopping method to curve-fit measure(s).

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
        ...     model=model, train_ds=train_ds, test_ds=test_ds, train_fn=train_fn, eval_fn=eval_fn, runs=3, substeps=5
        ... )
        >>> suff.evaluate()
        SufficiencyOutput(steps=array([  1,   3,  10,  31, 100], dtype=uint32), params={'test': array([ 0., 42.,  0.])}, measures={'test': array([1., 1., 1., 1., 1.])})
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
        params_output = get_curve_params(measures, ranges, niter)
        return SufficiencyOutput(ranges, params_output, measures)
