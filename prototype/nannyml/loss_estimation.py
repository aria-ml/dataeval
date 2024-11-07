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

def outputs_to_nannyml(problem_type, outputs, class_names, truths=None):
    classification = ("classification" in problem_type)
    has_labels = truths is not None
    pred_type = int if classification else float
    dict_out = {"y_pred": np.zeros(0, dtype=pred_type)}
    if has_labels:
        dict_out["y"] = np.zeros(0, dtype=pred_type)

        if len(truths) != len(outputs):
            raise ValueError(f"Length of truths {len(truths)} is",
                             f"not equal to length of outputs {len(outputs)}.")

    if classification:
        for class_name in class_names:
            dict_out[f"y_pred_proba_{class_name}"] = np.zeros(0)
    


    # Set model layers into evaluation mode
    #model.eval()
    #dataloader = DataLoader(dataset, batch_size=16)
    # Tell PyTorch to not track gradients, greatly speeds up processing
    for batch_id, output_batch in enumerate(outputs):
            if classification:
                preds = np.int64(torch.argmax(output_batch, dim=1))
                for i, class_name in enumerate(class_names):
                    key = f"y_pred_proba_{class_name}"
                    dict_out[key] = np.concatenate((dict_out[key], output_batch[:, i]))
            else:
                preds = output_batch[:,0]
            dict_out["y_pred"] = np.concatenate((dict_out["y_pred"], preds), dtype=pred_type)

            if has_labels:
                y_batch = truths[batch_id]
                if classification:
                    y = torch.Tensor(y_batch).int()
                else:
                    y = torch.Tensor(y_batch).float()
                dict_out["y"] = np.concatenate((dict_out["y"], y), dtype=pred_type)
            
    return pd.DataFrame(dict_out)


class LossEstimator():
    def __init__(self, problem_type="classification_multiclass", metrics=["accuracy"]):
        self.metrics = metrics
        self.problem_type = problem_type
        self.classification = "classification" in problem_type
        #self.metric = 'accuracy' if self.classification else 'rmse'
    
    def _eval_model(self, model: nn.Module, dataset: Dataset, class_names, has_labels: bool = False, device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, list]:
        #TODO: Let user set their own custom eval function
        pred_type = int if self.classification else float
        dict_out = {"y_pred": np.zeros(0, dtype=pred_type)}
        if has_labels:
            dict_out["y"] = np.zeros(0, dtype=pred_type)

        if self.classification:
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
                

                if self.classification:
                    preds = np.int64(torch.argmax(output, dim=1))
                    for i, class_name in enumerate(class_names):
                        key = f"y_pred_proba_{class_name}"
                        dict_out[key] = np.concatenate((dict_out[key], output[:, i]))
                else:
                    preds = output[:,0]
                dict_out["y_pred"] = np.concatenate((dict_out["y_pred"], preds), dtype=pred_type)

                if has_labels:
                    if self.classification:
                        y = torch.Tensor(batch[1]).int()
                    else:
                        y = torch.Tensor(batch[1]).float()
                    dict_out["y"] = np.concatenate((dict_out["y"], y), dtype=pred_type)
                

        return dict_out

    def evaluate(self, ref_df, op_df, class_names, chunk_size=50):
        if self.classification:
            y_pred_keys = {}
            for class_name in class_names:
                y_pred_keys[class_name] = f"y_pred_proba_{class_name}"
        else:
            feature_keys = []#set(class_names)

        est_kwargs = {
            #'problem_type': self.problem_type,
            #'y_pred_proba': y_pred_keys,
            'y_pred': 'y_pred',
            'y_true': 'y',
            'metrics': self.metrics,
            'chunk_size': chunk_size
        }

        if self.classification:
            estimator = nml.CBPE(problem_type=self.problem_type,
                                y_pred_proba = y_pred_keys,
                                **est_kwargs)
        else:
            estimator = nml.DLE(feature_column_names = feature_keys,
                **est_kwargs)
        

        estimator.fit(ref_df)
        results = estimator.estimate(op_df)

        reference_df = results.filter(period="reference").to_df()
        results_df = results.filter(period="analysis").to_df()
        
        has_drifted = False
        output_dict = {}
        for metric in self.metrics:
            ref_value = np.mean(reference_df[metric]['value'])
            pred_value = np.mean(results_df[metric]['value'])
            curr_alert = np.any(results_df[metric]['alert'])

            if curr_alert:
                has_drifted = True

            output_dict[f"Reference_{metric}"] = ref_value
            output_dict[f"Op_Predicted_{metric}"] = pred_value
            output_dict[f"Alert_{metric}"] = curr_alert

        output_dict["has_drifted"] = has_drifted


        return output_dict

    