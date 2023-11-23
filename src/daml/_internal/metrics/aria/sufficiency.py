from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from scipy.optimize import minimize

from daml._internal.datasets import DamlDataset


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


def create_data_indices(N: int, M: int) -> np.ndarray:
    """
    Randomly selects integers in range (0, M) an N number of times

    Parameters
    ----------
    N : int
        Size of the first dimension
    M : int
        Size of the second dimension

    Returns
    -------
    np.ndarray
        An NxM array of randomly selected integers in range (0, M)
    """
    return np.random.randint(0, M, size=(N, M))


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
    def __init__(
        self,
    ):
        # Used to calculate total number of training runs
        self._substeps = 0
        self._num_models = 0

        # Train & Eval functions must be set during run
        self._training_func = None
        self._eval_func = None

        self._output_dict = None

    def train(self, model, X, y, kwargs):
        if callable(self._training_func):
            self._training_func(model, X, y, **kwargs)

    def eval(self, model, X, y, kwargs):
        if callable(self._eval_func):
            return self._eval_func(model, X, y, **kwargs)

    def _set_func(self, func: Callable, error_msg="Argument was not a callable"):
        if callable(func):
            return func
        else:
            raise TypeError(error_msg)

    def set_training_func(self, func):
        self._training_func = self._set_func(func)

    def set_eval_func(self, func):
        self._eval_func = self._set_func(func)

    def setup(self, length, num_models, substeps):
        # Stores each models' metric output per step
        self._outputs = np.zeros((substeps, num_models))

        # Save the shape for plotting
        self._geomshape = (int(0.01 * length), length, substeps)
        self._ranges = np.geomspace(*self._geomshape).astype(int)

        # Randomly sample data indices for each model
        self._indices = create_data_indices(num_models, length)

    def run(
        self,
        model: nn.Module,
        train: DamlDataset,
        test: DamlDataset,
        train_kwargs: Dict[str, Any] = {},
        eval_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Creates data indices, trains models, and returns plotting data

        Parameters
        ----------
        train : DamlDataset
            Full training data that will be split for each run
        test : DamlDataset
            Data that will be used for every run's evaluation
        epochs : int
            Number of training cycles of the dataset, per model
        model_count : int
            Number of models to train and take averages from
        substeps : int
            Total number of dataset partitions that each model will train on

        Returns:
        Dict[str, Any]
        """

        # Convert datasets to correct type?
        X_test = torch.from_numpy(test.images.astype(np.float32))
        y_test = torch.from_numpy(test.labels)

        # For each model's dataset
        for j, inds in enumerate(self._indices):
            # Reset the network weights
            model = reset_parameters(model)

            # For each subset of data
            for i, substep in enumerate(self._ranges):
                # We warm start on new data
                b_inds = inds[:substep]
                X_train = torch.from_numpy(train.images[b_inds].astype(np.float32))
                y_train = torch.from_numpy(train.labels[b_inds])

                self.train(model, X_train, y_train, train_kwargs)

                # After each substep training, evaluate model on the test set
                self._outputs[i, j] = self.eval(model, X_test, y_test, eval_kwargs)

        n_i = self._ranges
        p_i = 1 - np.mean(self._outputs, axis=1)

        params = calc_params(p_i=p_i, n_i=n_i)

        self._output_dict = {
            "metric": self._outputs,
            "params": params,
            "n_i": n_i,
            "p_i": p_i,
            "geomshape": self._geomshape,
        }

        return self._output_dict

    def plot(self, output_dict: Dict[str, Any]):
        """Plotting function for data sufficiency tasks

        Parameters
        ----------
        params : np.ndarray
            The parameters used to calculate the line of best fit
        output_dict : Dict[str, Any]
            Output of sufficiency run

        """

        # Retrieve only relevant values in dictionary
        params = output_dict.get("params", None)
        n_i = output_dict.get("n_i", None)
        p_i = output_dict.get("p_i", None)
        geomshape = output_dict.get("geomshape", None)

        # Ensure all values were retrieved correctly
        if params is None:
            raise KeyError("params not found in output_dict")
        if n_i is None:
            raise KeyError("n_i not found in output_dict")
        if p_i is None:
            raise KeyError("p_i not found in output_dict")
        if geomshape is None:
            raise KeyError("geomshape not found in output_dict")

        # Setup 1 plt figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Accuracy vs Number of Images")
        ax.set_xlabel("Number of images")
        ax.set_ylabel("Accuracy")  # TODO: Possibly infer name based on given metric?
        ax.set_ylim(0, 1)

        # Plot model results
        ax.scatter(n_i, 1 - p_i, label="Model Results", s=15, c="black")

        # Plot line of best fit with extrapolation
        extrapolated = np.geomspace(geomshape[0], geomshape[1] * 4, geomshape[2])
        ax.plot(
            extrapolated,
            1 - f_out(extrapolated, params),
            linestyle="dashed",
            label="Potential Model Results",
        )

        ax.legend()
        # Save figure
        path = Path("Sufficiency Plot")
        self.save_fig(fig, path)  # type: ignore

    @staticmethod
    def save_fig(fig: Figure, path: Path):
        """
        Saves a `plt.figure` at a given path

        Parameters
        ---------
        fig : plt.figure
            Figure to be saved to png
        path : Path
            Location to save the figure
        Note
        ----
        A directory will be created if it does not exist
        """
        # TODO: Add folder creation, checking, and path handling
        fig.savefig(path)  # type: ignore
