from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy import stats
from tqdm import tqdm


class MetricEvaluator:

    def __init__(self, model_class, weight_files, validation_loader, test_loader, metric, device=None):
        self.model_class: nn.Module = model_class
        self.weight_files = weight_files
        self.val_loader = validation_loader
        self.test_loader = test_loader
        self.metric = metric
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")
    
    @torch.no_grad()
    def _compute_metrics_single_model(self, model):
        val_preds = []
        val_labels = []
        test_preds = []
        test_labels = []
        for val_img, val_label in self.val_loader:
            val_img = torch.stack(val_img).to(device=self.device)
            val_pred = model(val_img)
            val_preds.append(val_pred)
            val_labels.append(val_label)
        val_metrics = self.metric(val_preds, val_labels)
        for test_img, test_label in self.test_loader:
            test_img = torch.stack(test_img).to(device=self.device)
            test_pred = model(test_img)
            test_preds.append(test_pred)
            test_labels.append(test_label)
        test_metrics = self.metric(test_preds, test_labels)
        return val_metrics, test_metrics
    
    def load_weights(self, ckpt_file):
        model_instance = self.model_class
        checkpoint = torch.load(ckpt_file,weights_only=True)
        # This part is kind of janky and specific to how the model was defined by the user...
        state_dict = {k.lstrip("model."): v for k,v in checkpoint["state_dict"].items()}
        model_instance.load_state_dict(state_dict)
        model_instance.to(self.device)
        model_instance.eval()
        return model_instance
    
    def calculate_metrics(self, metric_dir=None, metric_filename=None):
        if not metric_dir:
            paths = [Path(wf) for wf in self.weight_files]
            common_dir = os.path.commonpath(paths)
            metric_dir = common_dir if len(common_dir) else paths[0].parent.parent
        if not metric_filename:
            metric_filename = "metric.np"
        self.val_filepath = f"{metric_dir}/val_{metric_filename}"
        self.test_filepath = f"{metric_dir}/test_{metric_filename}"
        val_metrics = np.zeros(len(self.weight_files))
        test_metrics = np.zeros(len(self.weight_files))
        for i,weight_file in enumerate(tqdm(self.weight_files, desc="Evaluating Models")):
            model = self.load_weights(weight_file)
            val_metric, test_metric = self._compute_metrics_single_model(model)
            val_metrics[i] = val_metric
            test_metrics[i] = test_metric
        np.save(self.val_filepath, val_metrics)
        np.save(self.test_filepath, test_metrics)

class MetricStressTester:

    def __init__(self, val_metric_file, test_metric_file, figure_savedir=None, figure_savename=None):
        self.val_metrics = np.load(val_metric_file)
        self.test_metrics = np.load(test_metric_file)
        if not figure_savedir:
            paths = [Path(wf) for wf in [val_metric_file, test_metric_file]]
            common_dir = os.path.commonpath(paths)
            figure_savedir = common_dir if len(common_dir) else paths[0].parent.parent
        if not figure_savename:
            figure_savename = "spearman_metric.png"
        self.figure_savedir = figure_savedir
        self.figure_savename = figure_savename    
 
    
    def spearman_permutation_test(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
        def statistic(v):
            rs = stats.spearmanr(v, y).statistic  # type: ignore
            transformed = rs 
            # transformed = rs * np.sqrt(dof / (1.0-rs**2))
            return transformed 
        dof = x.shape[0] - 2
        test_results = stats.permutation_test((x,), statistic, alternative='less',permutation_type='pairings')
        return {
            "statistic": test_results.statistic, # type: ignore
            "pvalue": test_results.pvalue, # type: ignore
            "null_distribution": test_results.null_distribution, # type: ignore
            "dof": dof,
        }
        
    def plot_confidence_interval(self, figure_name=None):
        if figure_name is None:
            figure_name = self.figure_savename
        null_hist = np.histogram(self.test_info["null_distribution"], bins=32, density=True)
        dist = stats.rv_histogram(null_hist, density=True)
        dist_vals = np.linspace(-1, 1, 32)
        pdf = dist.pdf(dist_vals)
        a, b = dist.interval(0.95, loc=self.test_info["statistic"])
        i = (a <= dist_vals) & (dist_vals <= b)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(dist_vals, pdf)
        ax.fill_between(dist_vals[i], 0, pdf[i], color="C0")
        ax.set_title("Spearman's Rank Correlation 95%% Confidence Interval")
        ax.set_xlabel("Statistic")
        ax.set_ylabel("Probability Density")
        plt.savefig(Path(self.figure_savedir)/figure_name, bbox_inches="tight")
        plt.close()

            
    def plot_distribution(self, figure_path):
        null_hist = np.histogram(self.test_info["null_distribution"], bins=32, density=True)
        dist = stats.rv_histogram(null_hist, density=True)
        dist_vals = np.linspace(-1, 1, 32)
        pdf = dist.pdf(dist_vals) # type: ignore
        pval = self.test_info["pvalue"]
        annotation = (f"p-value={pval:.3f}\n(shaded area)")
        props = dict(facecolor="black", width=1, headwidth=5, headlength=8)
        i = dist_vals < self.test_info["statistic"]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(dist_vals, pdf, label="PDF")
        ax.fill_between(dist_vals[i],y1=0,y2=pdf[i], color="C0")
        _ = ax.annotate(annotation, (-0.5, float(dist.pdf(-0.5)+0.05)), (-0.95, float(dist.pdf(-0.5)+0.1)), arrowprops=props)
        ax.set_title("Spearman's Rho Test Null Distribution")
        ax.set_xlabel("Statistic")
        ax.set_ylabel("Probability Density")
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()

    def calculate_underspecification(self):
        self.test_info = self.spearman_permutation_test(self.val_metrics, self.test_metrics)


def kappa_agreement_statistic(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    x_pred = np.argmax(x, axis=0)
    y_pred = np.argmax(y, axis=0)
    relative_agreement = (x_pred == y_pred).sum() / x_pred.shape[0]
    agreement_chance = 1/x.shape[1]
    kappa = (relative_agreement - agreement_chance) / (1 - agreement_chance)
    return kappa








