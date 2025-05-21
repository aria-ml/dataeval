from __future__ import annotations

__all__ = []

import contextlib
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure

from scipy.optimize import basinhopping

from dataeval.outputs._base import Output, set_metadata
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy


def f_out(n_i: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
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


def project_steps(params: NDArray[Any], projection: NDArray[Any]) -> NDArray[Any]:
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


def plot_measure(
    name: str,
    steps: NDArray[Any],
    measure: NDArray[Any],
    params: NDArray[Any],
    projection: NDArray[Any],
) -> Figure:
    import matplotlib.pyplot

    fig = matplotlib.pyplot.figure()
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


def f_inv_out(y_i: NDArray[Any], x: NDArray[Any]) -> NDArray[np.uint64]:
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


def inv_project_steps(params: NDArray[Any], targets: NDArray[Any]) -> NDArray[np.uint64]:
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


def calc_params(p_i: NDArray[Any], n_i: NDArray[Any], niter: int) -> NDArray[Any]:
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

    def is_valid(f_new, x_new, f_old, x_old) -> bool:  # noqa: ANN001
        return f_new != np.nan

    def f(x) -> float:  # noqa: ANN001
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


def get_curve_params(
    measures: Mapping[str, NDArray[Any]], ranges: NDArray[Any], niter: int
) -> Mapping[str, NDArray[Any]]:
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


@dataclass
class SufficiencyOutput(Output):
    """
    Output class for :class:`.Sufficiency` workflow.

    Attributes
    ----------
    steps : NDArray
        Array of sample sizes
    measures : Dict[str, NDArray]
        Average of values observed for each sample size step for each measure
    n_iter : int, default 1000
        Number of iterations to perform in the basin-hopping curve-fit process
    """

    steps: NDArray[np.uint32]
    measures: Mapping[str, NDArray[np.float64]]
    n_iter: int = 1000

    def __post_init__(self) -> None:
        c = len(self.steps)
        for m, v in self.measures.items():
            c_v = v.shape[1] if v.ndim > 1 else len(v)
            if c != c_v:
                raise ValueError(f"{m} does not contain the expected number ({c}) of data points.")
        self._params = None

    @property
    def params(self) -> Mapping[str, NDArray[Any]]:
        if self._params is None:
            self._params = {}
        if self.n_iter not in self._params:
            self._params[self.n_iter] = get_curve_params(self.measures, self.steps, self.n_iter)
        return self._params[self.n_iter]

    @set_metadata
    def project(
        self,
        projection: int | Iterable[int],
    ) -> SufficiencyOutput:
        """
        Projects the measures for each step.

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
        proj = SufficiencyOutput(projection, output, self.n_iter)
        proj._params = self._params
        return proj

    def plot(self, class_names: Sequence[str] | None = None) -> Sequence[Figure]:
        """
        Plotting function for data :term:`sufficience<Sufficiency>` tasks.

        Parameters
        ----------
        class_names : Sequence[str] | None, default None
            List of class names

        Returns
        -------
        Sequence[Figure]
            List of Figures for each measure

        Raises
        ------
        ValueError
            If the length of data points in the measures do not match

        Notes
        -----
        This method requires `matplotlib <https://matplotlib.org/>`_ to be installed.
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

    def inv_project(
        self, targets: Mapping[str, ArrayLike], n_iter: int | None = None
    ) -> Mapping[str, NDArray[np.float64]]:
        """
        Calculate training samples needed to achieve target model metric values.

        Parameters
        ----------
        targets : Mapping[str, ArrayLike]
            Mapping of target metric scores (from 0.0 to 1.0) that we want
            to achieve, where the key is the name of the metric.
        n_iter : int or None, default None
            Iteration to use when calculating the inverse power curve, if None defaults to 1000

        Returns
        -------
        Mapping[str, NDArray]
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
