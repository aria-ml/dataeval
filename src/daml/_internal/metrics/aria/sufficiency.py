from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Dataset, Subset

from daml._internal.metrics.outputs import SufficiencyOutput


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
    Retrieves the inverse power curve coefficients for the line of best fit

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
        inner = np.square(p_i - x[0] * n_i ** (-x[1]) - x[2])
        return np.sum(inner)

    res = minimize(
        f, np.array([0.5, 0.5, 0.1]), bounds=((0, None), (0, None), (0, None))
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


class Sufficiency:
    """
    Project dataset sufficiency using given a model and evaluation criteria
    """

    def __init__(
        self,
    ):
        # Train & Eval functions must be set during run
        self._training_func = None
        self._eval_func = None

    def _train(self, model: nn.Module, dataloader: DataLoader, kwargs: Dict[str, Any]):
        if self._training_func is None:
            raise TypeError("Training function is None. Set function before calling")

        self._training_func(model, dataloader, **kwargs)

    def _eval(
        self, model: nn.Module, dataloader: DataLoader, kwargs: Dict[str, Any]
    ) -> Dict[str, float]:
        if self._eval_func is None:
            raise TypeError("Eval function is None. Set function before calling")

        return self._eval_func(model, dataloader, **kwargs)

    def _set_func(self, func: Callable):
        if callable(func):
            return func
        else:
            raise TypeError("Argument was not a callable")

    def set_training_func(self, func: Callable):
        """
        Set the training function which will be executed each substep to train
        the provided model.

        Parameters
        ----------
        func : Callable[[torch.nn.Module, torch.utils.data.DataLoader], None]
            Function which takes a model (nn.Module) and a data loader (DataLoader)
            and executes model training against the data.
        """
        self._training_func = self._set_func(func)

    def set_eval_func(self, func: Callable):
        """
        Set the evaluation function which will be executed each substep
        in order to aggregate the resulting output for evaluation.

        Parameters
        ----------
        func : Callable[[torch.nn.Module, torch.utils.data.DataLoader], float]
            Function which takes a model (nn.Module) and a data loader (DataLoader)
            and returns a float which is used to assess model performance given
            the model and data.
        """
        self._eval_func = self._set_func(func)

    def run(
        self,
        model: nn.Module,
        train_ds: Dataset,
        test_ds: Dataset,
        runs: int,
        substeps: int,
        batch_size: int = 8,
        # Mutable sequences should not be used as default arg
        train_kwargs: Optional[Dict[str, Any]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> SufficiencyOutput:
        """
        Creates data indices, trains models, and returns plotting data

        Parameters
        ----------
        model : nn.Module
            Model that will be trained for each subset of data
        train_ds : Dataset
            Full training data that will be split for each run
        test_ds : Dataset
            Data that will be used for every run's evaluation
        runs : int
            Number of models to run over all subsets
        substeps : int
            Total number of dataset partitions that each model will train on
        batch_size : int, default 8
            The number of data points to be grouped during training and evaluation
        train_kwargs : Dict[str, Any] | None, default None
            Additional arguments required for custom training function
        eval_kwargs : Dict[str, Any] | None, default None
            Additional arguments required for custom evaluation function

        Returns
        -------
        SufficiencyOutput
            Dataclass containing the average of each measure per substep
        """

        if train_kwargs is None:
            train_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}

        if not hasattr(train_ds, "__len__"):
            raise TypeError("Must provide a dataset with a length attribute")
        length = getattr(train_ds, "__len__")()
        if length <= 0:
            raise ValueError("Length must be greater than 0")

        geomshape = (0.01 * length, length, substeps)  # Start, Stop, Num steps
        ranges = np.geomspace(*geomshape).astype(np.int64)

        # When given a new key (measure name), create an array of zeros as value
        metric_outputs = defaultdict(lambda: np.zeros(substeps))

        # Run each model over all indices
        for _ in range(runs):
            # Create a randomized set of indices to use
            indices = np.random.randint(0, length, size=length)
            # Reset the network weights to "create" an untrained model
            model = reset_parameters(model)
            # Run the model with each substep of data
            for iteration, substep in enumerate(ranges):
                # We warm start on new data
                subset = Subset(train_ds, indices[:substep])

                output = self._run_subset(
                    model,
                    subset,
                    test_ds,
                    batch_size,
                    train_kwargs,
                    eval_kwargs,
                )
                # Keep track of each measures values
                for name, value in output.items():
                    # Sum result into current substep iteration to be averaged later
                    metric_outputs[name][iteration] += value

        # The mean for each measure must be calculated before being returned
        mean_output = {name: 1 - (v / runs) for name, v in metric_outputs.items()}
        s = SufficiencyOutput(measures=mean_output, steps=ranges)

        return s

    def _run_subset(
        self,
        model: nn.Module,
        train_data: Dataset,
        eval_data: Dataset,
        batch_size: int,
        train_kwargs: Dict,
        eval_kwargs: Dict,
    ) -> Dict[str, float]:
        """Trains and evaluates model using custom functions"""
        train_loader = DataLoader(train_data, batch_size=batch_size)
        self._train(model, train_loader, train_kwargs)
        test_loader = DataLoader(eval_data, batch_size=batch_size)
        return self._eval(model, test_loader, eval_kwargs)

    def plot(self, data: SufficiencyOutput) -> List[Figure]:
        """Plotting function for data sufficiency tasks

        Parameters
        ----------
        data : SufficiencyOutput
            Dataclass containing the average of each measure per substep

        Returns
        -------
        List[plt.Figure]
            List of Figures for each measure

        """
        # X, y data
        X = data.steps
        measures: Dict[str, np.ndarray] = data.measures

        # Extrapolation parameters
        last_X = X[-1]
        geomshape = (0.01 * last_X, last_X * 4, len(X))
        extrapolated = np.geomspace(*geomshape).astype(np.int64)

        # Stores all plots
        plots = []

        # Create a plot for each measure on one figure
        for measure, values in measures.items():
            fig = plt.figure()
            fig = cast(Figure, fig)
            fig.tight_layout()

            ax = fig.add_subplot(111)

            ax.set_title(f"{measure} Sufficiency")
            ax.set_ylabel(f"{measure}")
            ax.set_xlabel("Steps")

            # Plot measure over each step
            ax.scatter(X, 1 - values, label="Model Results", s=15, c="black")

            # Plot extrapolation
            params = calc_params(p_i=values, n_i=X)
            ax.plot(
                extrapolated,
                1 - f_out(extrapolated, params),
                linestyle="dashed",
                label="Potential Model Results",
            )
            ax.legend()
            plots.append(fig)

        return plots
