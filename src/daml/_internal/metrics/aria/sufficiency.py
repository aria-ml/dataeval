from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.figure import Figure
from scipy.optimize import minimize
from torch.nn.modules.loss import _Loss as LossType
from torchmetrics import Metric

from daml._internal.datasets import DamlDataset


class Sufficiency:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: LossType,
        metric: Metric,
    ):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._metric = metric

    @staticmethod
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

    @staticmethod
    def get_params(p_i: np.ndarray, n_i: np.ndarray) -> np.ndarray:
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
            f, np.array([0.5, 0.5, 0.5]), bounds=((0, None), (0, None), (0, None))
        )

        return res.x

    @staticmethod
    def _create_data_indices(N: int, M: int) -> np.ndarray:
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

    @staticmethod
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

    def train_one_epoch(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor):
        """
        Passes data once through the model with backpropagation

        Parameters
        ----------
        model : nn.Module
            The trained model that will be evaluated
        X : torch.Tensor
            The training data to be passed through the model
        y : torch.Tensor
            The training labels corresponding to the data
        """

        # Zero out gradients
        self._optimizer.zero_grad()
        # Forward Propagation
        outputs = model(X)
        # Back prop
        loss = self._criterion(outputs, y)
        loss.backward()
        # Update optimizer
        self._optimizer.step()

    @staticmethod
    def eval(
        model: nn.Module, X: torch.Tensor, y: torch.Tensor, metric: Metric
    ) -> float:
        """
        Evaluate a model on a single pass with a given metric

        Parameters
        ----------
        model : nn.Module
            The trained model that will be evaluated
        X : torch.Tensor
            The testing data to be passed through th model
        y : torch.Tensor
            The testing labels corresponding to the data
        metric : Metric
            The statistic to calculate the performance of the model

        Returns
        -------
        float
            The calculated performance of the model
        """
        # Set model layers into evaluation mode
        model.eval()
        # Tell PyTorch to not track gradients, greatly speeds up processing
        with torch.no_grad():
            preds = model(X)
            metric.forward(preds, y)
            result = metric.compute()
        # Metrics accumulate data, so reset after each evaluation
        metric.reset()
        return result

    def run(
        self,
        train: DamlDataset,
        test: DamlDataset,
        epochs: int = 5,
        model_count: int = 3,
        substeps: int = 20,
        plot: bool = False,
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
        # Dataset metadata
        train_length = len(train)

        # Convert datasets to correct type?
        X_test = torch.from_numpy(test.images.astype(np.float32))
        y_test = torch.from_numpy(test.labels)

        accs = np.zeros((substeps, model_count))

        # Save the shape for plotting
        geomshape = (int(0.01 * train_length), train_length, substeps)
        ranges = np.geomspace(*geomshape).astype(int)

        # Randomly sample data indices for each model
        indices = self._create_data_indices(model_count, train_length)

        # For each model's dataset
        for j, inds in enumerate(indices):
            # Reset the network weights
            self._model = self.reset_parameters(self._model)

            # For each subset of data
            for i, substep in enumerate(ranges):
                # We warm start on new data
                b_inds = inds[:substep]
                X_train = torch.from_numpy(train.images[b_inds].astype(np.float32))
                y_train = torch.from_numpy(train.labels[b_inds])

                # Loop over the dataset multiple times
                for _ in range(epochs):
                    self.train_one_epoch(self._model, X_train, y_train)

                # After each substep training, evaluate model on the test set
                accs[i, j] = self.eval(self._model, X_test, y_test, self._metric)

        p_i = 1 - np.mean(accs, axis=1)
        n_i = ranges
        params = self.get_params(p_i=p_i, n_i=n_i)

        if plot:
            self.plot(params, n_i, p_i, geomshape=geomshape)

        output_dict = {
            "accuracy": accs,
            "params": params,
            "n_i": n_i,
            "p_i": p_i,
            "geomshape": geomshape,
        }

        return output_dict

    def plot(
        self, params: np.ndarray, n_i: np.ndarray, p_i: np.ndarray, geomshape: Tuple
    ):
        """Plotting function for data sufficiency tasks

        Parameters
        ----------
        params : np.ndarray
            The parameters used to calculate the line of best fit
        n_i : np.ndarray
            TODO: Ask Thayer
        p_i: np.ndarray
            TODO: Ask Thayer
        geomshape : Tuple
            The shape of all models' data
        """
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
            1 - self.f_out(extrapolated, params),
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
        fig.savefig(path)
