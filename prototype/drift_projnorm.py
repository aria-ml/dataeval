"""
This module contains the implementation of drift detection

"""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_weights(model: nn.Module):
    """
    Returns all of the model weights
    """
    weights = []
    for i, p in enumerate(model.parameters()):
        p_np = p.data.numpy().astype(np.float64)
        weights.append(p_np)

    return weights


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


class DatasetForcedLabels(Dataset):
    """
    Wrapper dataset, where the labels are computed at runtime by evaluating
    a model on the input. For example, an algorithm for computing a dataset
    dictates that we use a model to predict the labels on a dataset, and
    then fine-tune the model on those predicted labels. This serves as the
    dataset object to be used during that fine-tuning process.
    """

    # Force labels of old_dataset to be the output of model(old_dataset)
    def __init__(self, model, old_dataset):
        self.old_dataset = old_dataset
        self.model = model
        super().__init__()

    def __len__(self):
        return len(self.old_dataset)

    def __getitem__(self, idx):
        # Get the old input
        old_img, old_label = self.old_dataset[idx]

        # Evaluate a model on the old input
        # TODO: Calling the model here is inefficient.
        # This should be moved out of the innermost loop.
        new_logits = self.model(torch.unsqueeze(torch.Tensor(old_img), 0))

        # Set the label to the model's prediction from the old input
        new_label = new_logits.softmax(dim=1).argmax(dim=1)[0]
        return old_img, new_label


# todo: add cache (model, dataset) -> true/false underspecified
class DriftDetector:
    def __init__(
        self,
    ):
        # Train & Eval functions must be set during run
        self._training_func = None
        self._eval_func = None

        self.batch_size = -1
        self.train_kwargs = {}
        self.eval_kwargs = {}

    def _train(self, model: nn.Module, dataloader: DataLoader, kwargs: Dict[str, Any]):
        if self._training_func is None:
            raise TypeError("Training function is None. Set function before calling")

        self._training_func(model, dataloader, **kwargs)

    def _eval(self, model: nn.Module, dataloader: DataLoader, kwargs: Dict[str, Any]) -> Dict[str, float]:
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

    def set_train_eval_params(
        self,
        batch_size: int = 8,
        # Mutable sequences should not be used as default arg
        train_kwargs: Optional[Dict[str, Any]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if train_kwargs is None:
            train_kwargs = {}
        if eval_kwargs is None:
            eval_kwargs = {}
        self.batch_size = batch_size
        self.train_kwargs = train_kwargs
        self.eval_kwargs = eval_kwargs

    def run(
        self,
        model: nn.Module,
        train_ds: Dataset,
        labeled_test_datasets: List[Dataset],
        unlabeled_test_datasets: List[Dataset],
    ):
        """
        Predicts the effects of unlabeled test dataset drift on model accuracy.

        Use case: Given a model trained on some dataset, where we know its performance
        on a variety of labeled test datasets, how will model accuracy change on a new,
        possibly out-of-distribution test dataset that we don't have labels for?

        This implements the method introduced in https://arxiv.org/abs/2202.05834.

        Parameters
        ----------
        model : nn.Module
            The model that the effects of dataset drift will be characterized on.
        train_ds : Dataset
            Training dataset for the model.
        labeled_test_datasets : List[Dataset]
            A list of different test datasets, each of which has known labels.
            These will be used to approximate the effects of test dataset drift on
            model performance. The list must contain at least 2 datasets.
        unlabeled_test_datasets : List[Dataset]
            A list of different test datasets, each of which is unlabeled.
            The drift detector will predict the accuracy of the model on each of these
            datasets. If a labeled dataset is passed in, the labels will be ignored.

        Returns
        -------
        a_unknown : List[float]
            Array of the predicted model accuracies on each element of
            unlabeled_test_datasets.
            The ith entry in a_unknown corresponds to the ith entry in
            unlabeled_test_datasets.
        a_known : List[float]
            Array of the evaluated model accuracies on each element of
            labeled_test_datasets.
            The ith entry in a_known corresponds to the ith entry in
            labeled_test_datasets.
        """
        if len(labeled_test_datasets) < 2:
            raise Exception("Can't predict accuracy from fewer than two labeled test datasets")

        # We want to estimate the accuracy of the model on unlabeled_test_datasets.
        # The model accuracy on labeled_test_datasets is used to do this.

        # Fit a line to accuracy vs. projnorm for the provided labeled test data
        p_known = np.zeros(len(labeled_test_datasets))
        a_known = np.zeros(len(labeled_test_datasets))
        for i, test_ds in enumerate(labeled_test_datasets):
            pnorm, accuracy = self.get_projnorm_accuracy(
                model=model, train_ds=train_ds, test_ds=test_ds, labeled_test=True
            )

            p_known[i] = pnorm
            a_known[i] = accuracy

        # Build system of equations for projnorm/accuracy, solve for
        # (accuracy = pnorm * m + b)
        A = np.zeros((len(labeled_test_datasets), 2))
        A[:, 0] = p_known
        A[:, 1] = 1
        y = a_known

        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # Now that we fit a line to the projnorm/accuracy relationship, predict the
        # accuracies of the model on the unknown test datasets
        p_unknown = np.zeros(len(unlabeled_test_datasets))
        a_unknown = np.zeros(len(unlabeled_test_datasets))

        for i, test_ds in enumerate(unlabeled_test_datasets):
            pnorm, _ = self.get_projnorm_accuracy(model=model, train_ds=train_ds, test_ds=test_ds, labeled_test=False)
            p_unknown[i] = pnorm
            a_unknown[i] = m * pnorm + b

        return a_unknown, a_known

    # Work-in-progress: Check if a model is underspecified on the training data
    def _is_underspecified(
        self,
        model: nn.Module,
        train_ds: Dataset,
        underspec_runs: int = 0,
        # Mutable sequences should not be used as default arg
        train_kwargs: Optional[Dict[str, Any]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Placeholder function, for the time being. Just blanket return True to claim
        # that a model is always underspecified
        return True

        if not hasattr(train_ds, "__len__"):
            raise TypeError("Must provide a dataset with a length attribute")
        length = getattr(train_ds, "__len__")()
        if length <= 0:
            raise ValueError("Length must be greater than 0")

        # Check if model is underspecified. Currently unused.
        weights_underspecified = []
        # Run each model over all indices

        for _ in range(underspec_runs):
            # Create a randomized set of indices to use
            # indices = np.random.randint(0, length, size=length)
            # Reset the network weights to "create" an untrained model
            model = reset_parameters(model)
            # Run the model with each substep of data
            # for iteration, substep in enumerate(ranges):
            # We warm start on new data
            # subset = Subset(train_ds, indices[:substep])

            weights_curr = self._train_get_weights(model, train_ds)
            # Keep track of each measures values
            weights_underspecified.append(weights_curr)

        # Compute projnorm difference between each pair of weights, normalized to
        # weight magnitude
        weight_diffs = []
        for i in range(len(weights_underspecified)):
            for j in range(i, len(weights_underspecified)):
                pn_curr = self._compute_projnorm(weights_underspecified[i], weights_underspecified[j])
                pn_normalized = pn_curr / self._compute_weightnorm(weights_underspecified[i])
                weight_diffs.append(pn_normalized)

        weight_diffs = np.array(weight_diffs)
        # wd_mean = np.mean(weight_diffs)
        wd_stdev = np.std(weight_diffs)

        underspecified = wd_stdev > 0.2  # Threshold set arbitrarily, for now.
        return underspecified

    # Helper function for above, WIP
    def _compute_weightnorm(self, weights):
        mag_squared = 0
        for w_i in range(len(weights)):
            weights_flattened = weights[w_i].flatten()
            for w_1_i in range(len(weights_flattened)):
                mag_squared += (weights_flattened[w_1_i]) ** 2
        mag = np.sqrt(mag_squared)
        return mag

    def _weight_diff_norm_init(self, net_0, net_baseline):
        """
        # https://github.com/yaodongyu/ProjNorm/blob/main/projnorm.py
        Returns:
            the l2 norm difference the two networks
        """
        params1 = list(net_0.parameters())
        params2 = list(net_baseline.parameters())

        diff = 0
        for i in range(len(list(net_0.parameters()))):
            param1 = params1[i]
            param2 = params2[i]
            diff += (torch.norm(param1.flatten() - param2.flatten()) ** 2).cpu().detach().numpy()
        return np.sqrt(diff)

    def _compute_projnorm(self, weights_1, weights_2):
        """
        Computes the projection norm between two sets of model weights.

        The projection norm is described in detail in https://arxiv.org/abs/2202.05834.

        Parameters
        ----------
        weights_1 : list[np.array[float]]
            The weights from one model
        weights_2 : list[np.array[float]]
            The weights from another model

        Returns
        -------
        proj_norm : float
            The projection norm between weights_1 and weights_2
        """
        if len(weights_1) != len(weights_2):
            raise Exception("Tried to find projnorm of two sets of model weights of different sizes")
        # Iterate through all weights. proj_norm is the L2 distance between weights_1
        # and weights_2
        diff_squared = 0
        for w_i in range(len(weights_1)):
            weights_1_flattened = weights_1[w_i].flatten()
            weights_2_flattened = weights_2[w_i].flatten()
            for w_1_i in range(len(weights_1_flattened)):
                diff_squared += (weights_1_flattened[w_1_i] - weights_2_flattened[w_1_i]) ** 2
        proj_norm = np.sqrt(diff_squared)
        return proj_norm

    def get_projnorm_accuracy(
        self,
        model: nn.Module,
        train_ds: Dataset,
        test_ds: Dataset,
        labeled_test: bool,
        underspec_runs: int = 0,
    ):
        """
        Trains a reference model on a dataset, predicts labels on a test dataset, and
        fine-tunes the model on the predicted labels. From this, we obtain a projection
        norm, outlined in https://arxiv.org/abs/2202.05834.

        Parameters
        ----------
        model : nn.Module
            The model that the effects of dataset drift will be characterized on.
        train_ds : Dataset
            Training dataset for the model.
        test_ds : Dataset
            The dataset to predict labels on and fine-tune on.

        Returns
        -------
        proj_norm : float
            The projection norm associated with fine-tuning model on predicted labels
            on test_ds
        accuracy_ref : float
            If test_ds is labeled, this is the accuracy of the reference model
            on test_ds
        """

        if not hasattr(train_ds, "__len__"):
            raise TypeError("Must provide a dataset with a length attribute")
        length = getattr(train_ds, "__len__")()
        if length <= 0:
            raise ValueError("Length must be greater than 0")

        reference_model = reset_parameters(model)

        # Train the model on train_ds, and get its weights. This is the reference model.

        # weights_ref = self._train_get_weights(reference_model, train_ds)

        # Get the accuracy of the reference model on test_ds
        accuracy_ref = self._get_accuracy(reference_model, test_ds) if labeled_test else None

        # Create a new dataset, with inputs from test_ds and labels generated by
        # evaluating the model on test_ds

        # pred_ds = DatasetForcedLabels(reference_model, test_ds)

        # Fine-tune the model on this new dataset, and get its weights
        # We fine-tune a deepcopy so that we don't accidentally update reference_model,
        # since we use reference_model to lazily generate pred_ds
        finetuned_model = copy.deepcopy(reference_model)

        # weights_finetuned = self._train_get_weights(finetuned_model, pred_ds)

        # proj_norm = self._compute_projnorm(weights_ref, weights_finetuned)
        proj_norm = self._weight_diff_norm_init(reference_model, finetuned_model)

        return proj_norm, accuracy_ref

    def _train_get_weights(
        self,
        model: nn.Module,
        train_data: Dataset,
    ):
        """Trains and evaluates model using custom functions"""
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        self._train(model, train_loader, self.train_kwargs)
        weights = get_weights(model)
        return weights

    def _get_accuracy(self, model: nn.Module, eval_data: Dataset):
        test_loader = DataLoader(eval_data, batch_size=self.batch_size)
        eval_results = self._eval(model, test_loader, self.eval_kwargs)
        return eval_results["Accuracy"]
