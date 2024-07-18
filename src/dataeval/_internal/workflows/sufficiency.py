import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from scipy.optimize import basinhopping
from torch.utils.data import Dataset

from dataeval._internal.metrics.base import EvaluateMixin

STEPS_KEY = "_STEPS_"
PARAMS_KEY = "_CURVE_PARAMS_"

SufficiencyOutput = Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]


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


def f_inv_out(y_i: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Inverse function for f_out()

    Parameters
    ----------
    y_i : np.ndarray
        Data points for the line of best fit
    x : np.ndarray
        Array of inverse power curve coefficients

    Returns
    -------
    np.ndarray
        Array of sample sizes
    """
    n_i = ((y_i - x[2]) / x[0]) ** (-1 / x[1])
    return n_i


def calc_params(p_i: np.ndarray, n_i: np.ndarray, niter: int) -> np.ndarray:
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
    niter : int
        Number of iterations to perform in the basin-hopping
        numerical process to curve-fit p_i

    Returns
    -------
    np.ndarray
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


def validate_output(data: SufficiencyOutput):
    """Ensure the sufficiency data used is not malformed"""
    if not all(key in data for key in [STEPS_KEY, PARAMS_KEY]):
        raise KeyError(f"{STEPS_KEY} and {PARAMS_KEY} are required keys for Sufficiency output.")
    c = len(data[STEPS_KEY])
    for m, v in data.items():
        if m in [STEPS_KEY, PARAMS_KEY]:
            continue
        v = cast(np.ndarray, v)
        c_v = v.shape[1] if v.ndim > 1 else len(v)
        if c != c_v:
            raise ValueError("f{m} does not contain the expected number ({c}) of data points.")


def project_steps(params: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Projects the measures for each value of X

    Parameters
    ----------
    params : np.ndarray
        Inverse power curve coefficients used to calculate projection
    projection : np.ndarray
        Steps to extrapolate

    Returns
    -------
    np.ndarray
        Extrapolated measure values at each projection step

    """
    return 1 - f_out(projection, params)


def inv_project_steps(params: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Inverse function for project_steps()

    Parameters
    ----------
    params : np.ndarray
        Inverse power curve coefficients used to calculate projection
    targets : np.ndarray
        Desired measure values

    Returns
    -------
    np.ndarray
        Array of sample sizes, or 0 if overflow
    """
    steps = f_inv_out(1 - np.array(targets), params)
    steps[np.isnan(steps)] = 0
    return np.ceil(steps).astype(np.int64)


def get_curve_params(measures: Dict[str, np.ndarray], ranges: np.ndarray, niter: int) -> Dict[str, np.ndarray]:
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
    steps: np.ndarray,
    measure: np.ndarray,
    params: np.ndarray,
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
        project_steps(params, projection),
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
        eval_fn: Callable[[nn.Module, Dataset], Union[Dict[str, float], Dict[str, np.ndarray]]],
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
        value: Callable[[nn.Module, Dataset], Union[Dict[str, float], Dict[str, np.ndarray]]],
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

    def evaluate(self, eval_at: Optional[np.ndarray] = None, niter: int = 1000) -> SufficiencyOutput:
        """
        Creates data indices, trains models, and returns plotting data

        Parameters
        ----------
        eval_at : Optional[np.ndarray]
            Specify this to collect accuracies over a specific set of dataset lengths, rather
            than letting Sufficiency internally create the lengths to evaluate at.
        niter : int, default 1000
            Iterations to perform when using the basin-hopping method to curve-fit measure(s).

        Returns
        -------
        Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
            Dictionary containing the average of each measure per substep
        """
        if eval_at is not None:
            ranges = eval_at
        else:
            geomshape = (
                0.01 * self._length,
                self._length,
                self.substeps,
            )  # Start, Stop, Num steps
            ranges = np.geomspace(*geomshape).astype(np.int64)
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
                    indices[:substep].tolist(),
                    **self.train_kwargs,
                )

                # evaluate on test data
                measure = self.eval_fn(model, self.test_ds, **self.eval_kwargs)

                # Keep track of each measures values
                for name, value in measure.items():
                    if name in [STEPS_KEY, PARAMS_KEY]:
                        raise KeyError(f"Cannot use reserved name '{name}' as a metric name.")

                    # Sum result into current substep iteration to be averaged later
                    value = np.array(value).ravel()
                    if name not in measures:
                        measures[name] = np.zeros(substeps if len(value) == 1 else (substeps, len(value)))
                    measures[name][iteration] += value

        # The mean for each measure must be calculated before being returned
        measures = {k: (v / self.runs).T for k, v in measures.items()}
        params_output = get_curve_params(measures, ranges, niter)
        output = {STEPS_KEY: ranges, PARAMS_KEY: params_output}
        output.update(measures)
        return output

    @classmethod
    def project(
        cls,
        data: SufficiencyOutput,
        projection: Union[int, Sequence[int], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Projects the measures for each value of X

        Parameters
        ----------
        data : Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
            Dataclass containing the average of each measure per substep
        steps : Union[int, np.ndarray]
            Step or steps to project
        niter : int, default 200
            Number of iterations to perform in the basin-hopping
            numerical process to curve-fit data

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
        projection = np.array(projection) if isinstance(projection, Sequence) else projection
        if not isinstance(projection, np.ndarray):
            raise ValueError("'steps' must be an int, Sequence[int] or ndarray")

        output = {}
        output[STEPS_KEY] = projection
        for name, measures in data.items():
            if name in [STEPS_KEY, PARAMS_KEY]:
                continue
            measures = cast(np.ndarray, measures)
            if measures.ndim > 1:
                result = []
                for i in range(len(measures)):
                    projected = project_steps(data[PARAMS_KEY][name][i], projection)
                    result.append(projected)
                output[name] = np.array(result).T
            else:
                output[name] = project_steps(data[PARAMS_KEY][name], projection)
        return output

    @classmethod
    def plot(cls, data: SufficiencyOutput, class_names: Optional[Sequence[str]] = None) -> List[Figure]:
        """Plotting function for data sufficiency tasks

        Parameters
        ----------
        data : Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
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
        steps = cast(np.ndarray, data[STEPS_KEY])

        # Extrapolation parameters
        last_X = steps[-1]
        geomshape = (0.01 * last_X, last_X * 4, len(steps))
        extrapolated = np.geomspace(*geomshape).astype(np.int64)

        # Stores all plots
        plots = []

        # Create a plot for each measure on one figure
        for name, measures in data.items():
            if name in [STEPS_KEY, PARAMS_KEY]:
                continue
            measures = cast(np.ndarray, measures)
            if measures.ndim > 1:
                if class_names is not None and len(measures) != len(class_names):
                    raise IndexError("Class name count does not align with measures")
                for i, measure in enumerate(measures):
                    class_name = str(i) if class_names is None else class_names[i]
                    fig = plot_measure(
                        f"{name}_{class_name}",
                        steps,
                        measure,
                        data[PARAMS_KEY][name][i],
                        extrapolated,
                    )
                    plots.append(fig)

            else:
                fig = plot_measure(name, steps, measures, data[PARAMS_KEY][name], extrapolated)
                plots.append(fig)

        return plots

    @classmethod
    def inv_project(cls, targets: Dict[str, np.ndarray], data: SufficiencyOutput) -> Dict[str, np.ndarray]:
        """
        Calculate training samples needed to achieve target model metric values.

        Parameters
        ----------
        targets : Dict[str, np.ndarray]
            Dictionary of target metric scores (from 0.0 to 1.0) that we want
            to achieve, where the key is the name of the metric.

        data : Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
            Dataclass containing the average of each measure per substep

        Returns
        -------
        Dict[str, np.ndarray]
            List of the number of training samples needed to achieve each
            corresponding entry in targets
        """

        validate_output(data)

        projection = {}

        for name, target in targets.items():
            if name not in data:
                continue

            measure = cast(np.ndarray, data[name])
            if measure.ndim > 1:
                projection[name] = np.zeros((len(measure), len(target)))
                for i in range(len(measure)):
                    projection[name][i] = inv_project_steps(
                        data[PARAMS_KEY][name][i], target[i] if target.ndim == measure.ndim else target
                    )
            else:
                projection[name] = inv_project_steps(data[PARAMS_KEY][name], target)

        return projection
