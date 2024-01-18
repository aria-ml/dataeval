from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from scipy.optimize import minimize
from torch.utils.data import DataLoader

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
        # Train & Eval functions must be set during run
        self._training_func = None
        self._eval_func = None

        self._output_dict = None

    def _train(self, model: nn.Module, dataloader: DataLoader, kwargs: Dict[str, Any]):
        if self._training_func is None:
            raise TypeError("Training function is None. Set function before calling")

        self._training_func(model, dataloader, **kwargs)

    def _eval(self, model: nn.Module, dataloader: DataLoader, kwargs: Dict[str, Any]):
        if self._eval_func is None:
            raise TypeError("Eval function is None. Set function before calling")

        return self._eval_func(model, dataloader, **kwargs)

    def _set_func(self, func: Callable, error_msg="Argument was not a callable"):
        if callable(func):
            return func
        else:
            raise TypeError(error_msg)

    def set_training_func(self, func: Callable):
        self._training_func = self._set_func(func)

    def set_eval_func(self, func: Callable):
        self._eval_func = self._set_func(func)

    def setup(self, length: int, num_models: int = 1, substeps: int = 1):
        if length <= 0:
            raise ValueError("Length cannot be 0")

        # Stores each models' metric output per step
        self._outputs = np.zeros((substeps, num_models))
        # Save the shape for plotting
        self._geomshape = (0.01 * length, length, substeps)
        self._ranges = np.geomspace(*self._geomshape).astype(np.int64)

        # Randomly sample data indices for each model
        self._indices = create_data_indices(num_models, length)

    def run(
        self,
        model: nn.Module,
        train_ds: DamlDataset,
        test_ds: DamlDataset,
        batch_size: int = 8,
        train_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # Mutable sequences should not be used as default arg
        eval_kwargs: Optional[Dict[str, Any]] = None,
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

        if train_kwargs is None:
            train_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}

        # BUG -> Manual conversion required. PyTorch fails on non np.float32
        X_test = test_ds.images.astype(np.float32)
        y_test = test_ds.labels
        # Bootstrapping
        for j, inds in enumerate(self._indices):
            # Reset the network weights
            model = reset_parameters(model)

            # For each subset of data
            for i, substep in enumerate(self._ranges):
                # We warm start on new data
                b_inds = inds[:substep]

                images: np.ndarray = train_ds.images[b_inds].astype(np.float32)
                labels = train_ds.labels[b_inds] if len(train_ds.labels) else None
                boxes = train_ds.boxes[b_inds] if len(train_ds.boxes) else None

                subset_dataset = DamlDataset(images, labels, boxes)
                subset_test = DamlDataset(X_test, y_test)

                self._outputs[i, j] = self._run_subset(
                    model,
                    subset_dataset,
                    subset_test,
                    batch_size,
                    train_kwargs,
                    eval_kwargs,
                )

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

    def _run_subset(
        self,
        model: nn.Module,
        train_data: DamlDataset,
        eval_data: DamlDataset,
        batch_size: int,
        train_kwargs,
        eval_kwargs,
    ):
        train_loader = DataLoader(train_data, batch_size=batch_size)
        self._train(model, train_loader, train_kwargs)
        test_loader = DataLoader(eval_data, batch_size=batch_size)
        return self._eval(model, test_loader, eval_kwargs)

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
        extrapolated = np.geomspace(
            geomshape[0], geomshape[1] * 4, geomshape[2]
        ).astype(np.int64)
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
