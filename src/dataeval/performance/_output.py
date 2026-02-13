__all__ = []

import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import basinhopping

from dataeval.protocols import ArrayLike
from dataeval.types import DictOutput
from dataeval.utils.arrays import as_numpy

_logger = logging.getLogger(__name__)

LRU_CACHE_SIZE = 10


@dataclass
class Constraints:
    """
    Dataclass containing constraints for the coefficients of the power law: f[n] = c * n**(-m) + c0.

    Attributes
    ----------
    scale : tuple
        high and low constraints for coefficient c
    negative_exponent : tuple
        high and low constraints for exponent m
    asymptote : tuple
        high and low constraints for asymptote c0
    """

    scale: tuple[None, None] = (None, None)
    negative_exponent: tuple[Literal[0], None] = (0, None)
    asymptote: tuple[(Literal[0] | None), (Literal[1] | None)] = (None, None)
    """
    returns list of coefficient constraints
    """

    def to_list(self) -> list[tuple]:
        return [self.scale, self.negative_exponent, self.asymptote]


@dataclass
class SufficiencyOutput(DictOutput):
    """
    Output class for :class:`.Sufficiency` workflow.

    Attributes
    ----------
    steps : NDArray
        Array of sample sizes
    measures : dict[str, NDArray]
        3D array [runs, substep, classes] of values for all runs observed for each sample size step for each measure
    averaged_measures : dict[str, NDArray]
        Average of values for all runs observed for each sample size step for each measure
    unit_interval : bool, default True
        Constrains the power law to the interval [0, 1].  Set True (default) for metrics such as accuracy, precision,
        and recall which are defined to take values on [0,1].  Set False for metrics not on the unit interval.
    """

    steps: NDArray[np.intp]
    measures: Mapping[str, NDArray[Any]]
    averaged_measures: MutableMapping[str, NDArray[Any]] = field(default_factory=lambda: {})
    unit_interval: bool = True

    def __post_init__(self) -> None:
        if len(self.averaged_measures) == 0:
            for metric, values in self.measures.items():
                self.averaged_measures[metric] = np.asarray(np.mean(values, axis=0)).T
        c = len(self.steps)
        for m, v in self.averaged_measures.items():
            c_v = v.shape[1] if v.ndim > 1 else len(v)
            if c != c_v:
                raise ValueError(f"{m} does not contain the expected number ({c}) of data points.")
        self._params: dict[int, Mapping[str, NDArray[np.float64]]] | None = None
        self._params_cache_keys: list[int] = []

    @property
    def params(self) -> Mapping[str, NDArray[np.float64]]:
        """
        Get the curve fit parameters for the power law model.

        Returns
        -------
        Mapping[str, NDArray]
            Mapping of metric names to their fitted power law parameters
        """
        return self.get_params()

    def get_params(self, n_iter: int = 1000) -> Mapping[str, NDArray[np.float64]]:
        """
        Get the curve fit parameters for the power law model.

        Parameters
        ----------
        n_iter : int, default 1000
            Number of iterations to perform in the basin-hopping curve-fit process

        Returns
        -------
        Mapping[str, NDArray]
            Mapping of metric names to their fitted power law parameters
        """
        if self._params is None:
            self._params = {}

        if n_iter in self._params:
            # Move to end (most recently used)
            if n_iter in self._params_cache_keys:
                self._params_cache_keys.remove(n_iter)
            self._params_cache_keys.append(n_iter)
        else:
            # Evict oldest if cache is full
            if len(self._params_cache_keys) >= LRU_CACHE_SIZE:
                oldest = self._params_cache_keys.pop(0)
                del self._params[oldest]
            # Compute and cache
            self._params[n_iter] = get_curve_params(self.averaged_measures, self.steps, n_iter, self.unit_interval)
            self._params_cache_keys.append(n_iter)

        return self._params[n_iter]

    def project(self, projection: int | Iterable[int], n_iter: int = 1000) -> pl.DataFrame:
        """
        Project metric values to new sample sizes using the fitted power law.

        Parameters
        ----------
        projection : int | Iterable[int]
            Sample size(s) to project metric values to
        n_iter : int, default 1000
            Number of iterations to perform in the basin-hopping curve-fit process

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
                step: int - The projected sample size
                <metric>: float - One column per metric (multi-class as metric_0, metric_1, ...)

        Raises
        ------
        ValueError
            If `projection` is not numerical
        """
        proj_array = np.asarray(list(projection) if isinstance(projection, Iterable) else [projection])
        if not np.issubdtype(proj_array.dtype, np.number):
            raise ValueError("'projection' must consist of numerical values")

        params = self.get_params(n_iter)
        output: dict[str, NDArray[Any]] = {}
        for name, averaged_measures in self.averaged_measures.items():
            if averaged_measures.ndim > 1:
                result = []
                for i in range(len(averaged_measures)):
                    projected = project_steps(params[name][i], proj_array)
                    result.append(projected)
                output[name] = np.array(result)
            else:
                output[name] = project_steps(params[name], proj_array)

        return self._build_dataframe(proj_array, output)

    def inv_project(self, targets: Mapping[str, ArrayLike] | ArrayLike, n_iter: int = 1000) -> pl.DataFrame:
        """
        Calculate training samples needed to achieve target metric values.

        Parameters
        ----------
        targets : Mapping[str, ArrayLike] | ArrayLike
            Mapping of target metric scores (from 0.0 to 1.0) that we want
            to achieve, where the key is the name of the metric.  If an
            ArrayLike is provided, the same targets are applied to all metrics.
        n_iter : int, default 1000
            Number of iterations to perform in the basin-hopping curve-fit process

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
                target: float - Target metric value
                <metric>: int - One column per metric (multi-class as metric_0, metric_1, ...)
                Values of -1 indicate the target is unachievable.
        """
        params = self.get_params(n_iter)
        results: dict[str, tuple[NDArray[Any], NDArray[np.int64]]] = {}  # metric -> (targets, samples)

        if not isinstance(targets, Mapping):
            targets = dict.fromkeys(self.averaged_measures.keys(), targets)

        for name, target in targets.items():
            tarray = as_numpy(target)
            if name not in self.averaged_measures:
                continue

            measure = self.averaged_measures[name]
            if measure.ndim > 1:
                samples = np.zeros((len(measure), len(tarray)), dtype=np.int64)
                for i in range(len(measure)):
                    samples[i] = inv_project_steps(
                        params[name][i],
                        tarray[i] if tarray.ndim == measure.ndim else tarray,
                    )
                results[name] = (tarray, samples)
            else:
                results[name] = (tarray, inv_project_steps(params[name], tarray))

        return self._build_inv_project_dataframe(results)

    @staticmethod
    def _build_inv_project_dataframe(
        results: Mapping[str, tuple[NDArray[Any], NDArray[np.int64]]],
    ) -> pl.DataFrame:
        """Build wide-format DataFrame from inverse projection results."""
        # Collect all unique target values
        all_targets: set[float] = set()
        for targets, _ in results.values():
            for t in targets.flatten() if targets.ndim > 1 else targets:
                all_targets.add(float(t))

        sorted_targets = sorted(all_targets)
        data: dict[str, list[Any]] = {"target": sorted_targets}

        for name, (targets, samples) in results.items():
            if samples.ndim > 1:
                for class_idx, class_samples in enumerate(samples):
                    t_vals = targets[class_idx] if targets.ndim > 1 else targets
                    target_to_sample = {float(t_vals[i]): int(class_samples[i]) for i in range(len(class_samples))}
                    data[f"{name}_{class_idx}"] = [target_to_sample.get(t) for t in sorted_targets]
            else:
                target_to_sample = {float(targets[i]): int(samples[i]) for i in range(len(samples))}
                data[name] = [target_to_sample.get(t) for t in sorted_targets]

        return pl.DataFrame(data).sort("target")

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert averaged measures to a DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with step column and one column per metric.
        """
        return self._build_dataframe(self.steps, self.averaged_measures)

    @staticmethod
    def _build_dataframe(
        steps: NDArray[np.intp],
        measures: Mapping[str, NDArray[Any]],
    ) -> pl.DataFrame:
        """Build wide-format DataFrame from steps and measures."""
        data: dict[str, list[Any]] = {"step": [int(s) for s in steps]}
        for name, measure in measures.items():
            if measure.ndim > 1:
                for class_idx, class_data in enumerate(measure):
                    data[f"{name}_{class_idx}"] = [float(v) for v in class_data]
            else:
                data[name] = [float(v) for v in measure]
        return pl.DataFrame(data).sort("step")

    @property
    def plot_type(self) -> Literal["sufficiency"]:
        return "sufficiency"


def f_out(n_i: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
    """
    Calculate the line of best fit based on its free parameters.

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


def project_steps(params: NDArray[Any], projection: NDArray[Any]) -> NDArray[Any]:
    """Projects the measures for each value of X.

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


def f_inv_out(y_i: NDArray[Any], x: NDArray[Any]) -> NDArray[np.int64]:
    """
    Inverse function for f_out().

    Parameters
    ----------
    y_i : NDArray
        Data points for the line of best fit
    x : NDArray
        Array of inverse power curve coefficients

    Returns
    -------
    NDArray
        Sample size or -1 if unachievable for each data point
    """
    with np.errstate(invalid="ignore"):
        n_i = ((y_i - x[2]) / x[0]) ** (-1 / x[1])
    unachievable_targets = np.isnan(n_i) | np.any(n_i > np.iinfo(np.int64).max)
    if any(unachievable_targets):
        with np.printoptions(suppress=True):
            _logger.warning(
                "Number of samples could not be determined for target(s): "
                f"""{
                    np.array2string(
                        1 - y_i[unachievable_targets],
                        separator=", ",
                        formatter={"float": lambda x: f"{x}"},
                    )
                }"""
                " with asymptote of " + str(1 - x[2]),
            )
        n_i[unachievable_targets] = -1
    return np.asarray(n_i, dtype=np.int64)


def inv_project_steps(params: NDArray[Any], targets: NDArray[Any]) -> NDArray[np.int64]:
    """Inverse function for project_steps().

    Parameters
    ----------
    params : NDArray
        Inverse power curve coefficients used to calculate projection
    targets : NDArray
        Desired measure values

    Returns
    -------
    NDArray
        Samples required or -1 if unachievable for each target value
    """
    steps = f_inv_out(1 - np.array(targets), params)
    return np.ceil(steps)


def linear_initialization(metric: NDArray[Any], sizes: NDArray[Any], bounds: Constraints) -> NDArray[np.float64]:
    """
    Linear initialization of x for power law: f[n] = x[0] * n**(-x[1]) + x[2].

    Parameters
    ----------
    metric : NDArray[Any]
        Array of metric values
    sizes : NDArray[Any]
        Array of sample sizes
    bounds : list[Any]
        List of tuples representing bounds for each parameter
    """
    with np.errstate(all="raise"):
        try:
            # determine whether this is an increasing or decreasing power law
            diff = np.sum(np.diff(metric))
            asymptote = y = 0
            x = np.log(sizes)

            if diff > 0:
                asymptote = np.max(metric) + 0.001
                y = np.log(asymptote - metric)

            else:
                asymptote = np.min(metric) - 0.001
                y = np.log(metric - asymptote)

            negative_exponent, intercept = np.polyfit(x, y, 1)
            # Check c0 bounds and constrain accordingly after
            scale = np.exp(intercept) if diff <= 0 else -np.exp(intercept)
            if bounds.asymptote == (0, 1):
                asymptote = max(0, min(asymptote, 1))
        except (ArithmeticError, ValueError, RuntimeWarning):
            # encountering error in linear initialization will default to hardcoded guess
            _logger.warning("Error applying linear initialization for initial guess, using default")
            scale = 0.5
            negative_exponent = -0.5
            asymptote = 1
    return np.array([scale, -negative_exponent, asymptote], dtype=np.float64)


def calc_params(p_i: NDArray[Any], n_i: NDArray[Any], niter: int, unit_interval: bool) -> NDArray[np.float64]:
    """
    Retrieve the inverse power curve coefficients for the line of best fit.

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
    bounds : Constraints
        dataclass containing constraints for each parameter [scale, negative_exponent, asymptote]

    Returns
    -------
    NDArray
        Array of parameters to recreate line of best fit
    """
    bounds = Constraints(
        scale=(None, None),
        negative_exponent=(0, None),
        asymptote=((0, 1) if (unit_interval) else (None, None)),
    )

    def is_valid(
        bounds: Constraints,
    ) -> Callable[[float, NDArray[np.floating], float, NDArray[np.floating]], bool]:
        def accept_test(f_new, x_new, f_old, x_old) -> bool:  # noqa: ANN001, ARG001
            constraints_list = bounds.to_list()
            in_bounds = all(
                low is None or low <= val and high is None or high >= val
                for val, (low, high) in zip(x_new, constraints_list, strict=False)
            )
            is_nan = np.isnan(f_new)
            if not in_bounds or is_nan:
                _logger.log(
                    logging.INFO,
                    f"Minimizer Attempted to Step to Minimum {f_new} with parameters {x_new}"
                    f" and bounds of {constraints_list}",
                    {"f_new": f_new, "parameters": x_new, "bounds": constraints_list},
                )
            return in_bounds and not is_nan

        return accept_test

    def f(x) -> float:  # noqa: ANN001
        try:
            return np.sum(np.square(p_i - f_out(n_i, x)))
        except RuntimeWarning:
            return np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        res = basinhopping(
            f,
            x0=linear_initialization(p_i, n_i, bounds),
            niter=niter,
            stepsize=1.0,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds.to_list()},
            accept_test=is_valid(bounds),
            niter_success=200,
        )
    return res.x


def get_curve_params(
    averaged_measures: MutableMapping[str, NDArray[Any]],
    ranges: NDArray[Any],
    niter: int,
    unit_interval: bool,
) -> Mapping[str, NDArray[np.float64]]:
    """Calculate and aggregate parameters for both single and multiclass metrics."""
    output = {}
    for name, measure in averaged_measures.items():
        measure = cast(np.ndarray, measure)
        if measure.ndim > 1:
            result = [calc_params(1 - value, ranges, niter, unit_interval) for value in measure]
            output[name] = np.array(result)
        else:
            output[name] = calc_params(1 - measure, ranges, niter, unit_interval)
    return output
