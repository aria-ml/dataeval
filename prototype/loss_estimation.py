import os
import random
from typing import Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset
import nannyml as nml
from IPython.display import display

class LossEstimator():
    def __init__(self, method="CBPE", problem_type="classification_multiclass"):
        self.method = method
        self.problem_type = problem_type
    
    def _eval_model(self, model: nn.Module, dataset: Dataset, class_names, has_labels: bool = False, device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, list]:
        #TODO: Let user set their own custom eval function
        dict_out = {"y_pred": np.zeros(0, dtype=int)}
        if has_labels:
            dict_out["y"] = np.zeros(0, dtype=int)

        for class_name in class_names:
            dict_out[f"y_pred_proba_{class_name}"] = np.zeros(0)

        # Set model layers into evaluation mode
        model.eval()
        dataloader = DataLoader(dataset, batch_size=16)
        # Tell PyTorch to not track gradients, greatly speeds up processing
        with torch.no_grad():
            for batch in dataloader:
                # Load data/images to device
                X = torch.Tensor(batch[0]).to(device)
                # Load targets/labels to device
                
                output = model(X).cpu()
                preds = np.int64(torch.argmax(output, dim=1))

                dict_out["y_pred"] = np.concatenate((dict_out["y_pred"], preds), dtype=int)

                if has_labels:
                    y = torch.Tensor(batch[1]).int()
                    dict_out["y"] = np.concatenate((dict_out["y"], y), dtype=int)
                for i, class_name in enumerate(class_names):
                    key = f"y_pred_proba_{class_name}"
                    dict_out[key] = np.concatenate((dict_out[key], output[:, i]))

        return dict_out

    def evaluate(self, model, ref_data, op_data, class_names, chunk_size=50, eval_kwargs={}):

        ref_dict = self._eval_model(model, ref_data, class_names, True, **eval_kwargs)
        ref_df = pd.DataFrame(ref_dict)

        op_dict = self._eval_model(model, op_data, class_names, False, **eval_kwargs)
        op_df = pd.DataFrame(op_dict)

        y_pred_keys = {}
        for class_name in class_names:
            y_pred_keys[class_name] = f"y_pred_proba_{class_name}"

        estimator = nml.CBPE(
        problem_type=self.problem_type,
        y_pred_proba=y_pred_keys,
        y_pred="y_pred",
        y_true="y",
        metrics=["accuracy"],
        chunk_size=chunk_size,
    )
        

        estimator.fit(ref_df)
        results = estimator.estimate(op_df)

        reference_df = results.filter(period="reference").to_df()
        results_df = results.filter(period="analysis").to_df()
        ref_accuracy = np.mean(reference_df['accuracy']['value'])
        pred_accuracy = np.mean(results_df['accuracy']['value'])
        alert = np.any(results_df['accuracy']['alert'])

        return {
            "Reference_Accuracy": ref_accuracy,
            "Op_Predicted_Accuracy": pred_accuracy,
            "Has_Drifted": alert
        }