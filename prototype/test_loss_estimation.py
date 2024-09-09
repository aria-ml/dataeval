import pytest
from loss_estimation import LossEstimator, outputs_to_nannyml
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
import torchmetrics
from torch.utils.data import DataLoader, Dataset, Subset
import nannyml as nml
from IPython.display import display
from PIL import Image


def custom_train_class(model: nn.Module, dataset: Dataset, device, epochs=10):
    # Defined only for this testing scenario
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Define the dataloader for training
    dataloader = DataLoader(dataset, batch_size=16)

    for epoch in range(epochs):
        for batch in dataloader:
            # Load data/images to device
            X = torch.Tensor(batch[0]).to(device)
            # Load targets/labels to device
            y = torch.Tensor(batch[1]).to(device)
            # Zero out gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(X)
            # Compute loss
            loss = criterion(outputs, y)
            # Back prop
            loss.backward()
            # Update weights/parameters
            optimizer.step()

def custom_train_regress(model: nn.Module, dataset: Dataset, device, epochs=10):
    # Defined only for this testing scenario
    criterion = torch.nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Define the dataloader for training
    dataloader = DataLoader(dataset, batch_size=16)

    for epoch in range(epochs):
        for batch in dataloader:
            # Load data/images to device
            X = torch.Tensor(batch[0]).to(device)
            # Load targets/labels to device
            y = torch.Tensor(batch[1]).to(device)
            # Zero out gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(X)
            # Compute loss
            loss = criterion(outputs.float(), y.float())
            # Back prop
            loss.backward()
            # Update weights/parameters
            optimizer.step()

def custom_eval(model: nn.Module, dataset: Dataset, class_names, has_labels: bool = False, device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, list]:
        outputs = []
        labels = []

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
                outputs.append(output)
                if has_labels:
                    labels.append(batch[1])
                

        return outputs, labels

def get_accuracy(model: nn.Module, dataset: Dataset, class_names, device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, list]:
    # metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    # result = 0
    # batch_dicts = []
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)



    # Set model layers into evaluation mode
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16)
    # Tell PyTorch to not track gradients, greatly speeds up processing
    with torch.no_grad():
        for batch in dataloader:
            # Load data/images to device
            X = torch.Tensor(batch[0]).to(device)
            # Load targets/labels to device
            y = torch.Tensor(batch[1]).int()
            output = model(X).cpu()

            metric.update(output, y)
        result = metric.compute().cpu()
    return {"Accuracy": result}

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

class MockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10*10*3, 84)
        self.fc2 = nn.Linear(84, 3)
        self.softmax = torch.nn.Softmax()
    
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class RegressNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10*10*3, 84)
        self.fc2 = nn.Linear(84, 3)
        self.fc3 = nn.Linear(3, 1)
    
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SimpleDataset(datasets.VisionDataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images = images
        self.labels = labels
        self.class_names = sorted(np.unique(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).float()

        label = torch.tensor(self.labels[idx])

        return img, label

class MockDataset(SimpleDataset):
    def __init__(self):
        images = [
            np.zeros((3,10,10)),
            np.ones((3,10,10)),
            2*np.ones((3,10,10)),
            np.zeros((3,10,10)),
            255*np.ones((3,10,10)),
            252*np.ones((3,10,10)),
            
        ]
        labels = [
            1,
            2,
            2,
            1,
            0,
            0
        ]
        super().__init__(images, labels)

class LargeMockDataset(SimpleDataset):
    def __init__(self):
        # sklearn fails when dataset is too small due to "test_size=1" error
        # more info in https://github.com/interpretml/interpret/issues/142
        labels = [0,1,2] * 10
        images = []
        for i, label in enumerate(labels):
            images.append(label * np.ones((3,10,10)))
        super().__init__(images, labels)
    

class TestLEUnit():
    def test_eval_builds_dict(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        ds = MockDataset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(MockNet().to(device))
        model = cast(MockNet, model)
        le = LossEstimator()

        #ds_dict = le._eval_model(model, ds, ds.class_names, True)
        ds_outputs, ds_labels = custom_eval(model, ds, ds.class_names, True)
        ds_dict = outputs_to_nannyml("classification", ds_outputs, ds.class_names, ds_labels)
        assert sorted(ds_dict["y"]) == sorted(ds.labels)
    
    def test_fails_with_bad_truths_len(self):
        problem_type = "classification"
        outputs = [1,2,3]
        class_names = [1,2,3]
        truths = [1,2,3,1]
        with pytest.raises(ValueError):
            _ = outputs_to_nannyml(problem_type, outputs, class_names, truths)

class TestLEFunc_class():
    def test_no_degradation_on_same_dataset_class(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        ds = LargeMockDataset()
        ds_c = LargeMockDataset()
        class_names = np.unique(ds.labels)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(MockNet().to(device))
        model = cast(MockNet, model)

        model = reset_parameters(model)
        # Run the model with each substep of data
        # train on subset of train data
        custom_train_class(
            model,
            ds,
            device
        )

        #for method in ("regression", "classification_multiclass"):
        method = "classification_multiclass"
        metrics = ["accuracy"]
        estimator = LossEstimator(method, metrics)
        #results = estimator.evaluate(model, ds, ds_c, ds.class_names, 2)
        ds_c_outputs, _ = custom_eval(model, ds_c, ds_c.class_names, False)
        ds_c_dict = outputs_to_nannyml("classification", ds_c_outputs, ds_c.class_names)

        ds_outputs, ds_labels = custom_eval(model, ds, ds.class_names, True)
        ds_dict = outputs_to_nannyml("classification", ds_outputs, ds.class_names, ds_labels)

        results = estimator.evaluate(ds_dict, ds_c_dict, class_names, 2)

        ds_acc = results["Reference_accuracy"]
        ds_c_acc = results["Op_Predicted_accuracy"]

        assert np.isclose(ds_acc, ds_c_acc)
    
    def inactive_test_no_degradation_on_same_dataset_regress(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        ds = LargeMockDataset()
        ds_c = LargeMockDataset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(RegressNet().to(device))
        model = cast(RegressNet, model)

        model = reset_parameters(model)
        # Run the model with each substep of data
        # train on subset of train data

        custom_train_regress(
            model,
            ds,
            device
        )

        #for method in ("regression", "classification_multiclass"):
        method = "regression"
        metrics = ["rmse"]
        estimator = LossEstimator(method, metrics)
        results = estimator.evaluate(model, ds, ds_c, ds.class_names, 2)

        ds_acc = results["Reference_accuracy"]
        ds_c_acc = results["Op_Predicted_accuracy"]

        assert np.isclose(ds_acc, ds_c_acc)

    def test_degrades_on_corrupted_dataset(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True
        np.random.seed(0)

        ds = LargeMockDataset()
        ds_c = LargeMockDataset()
        
        for i, img in enumerate(ds_c.images):
            #img[0,:,:] -= 1.5
            #img[1,:,:] += 1.5
            #img[:,:,:] + 
            #x = img / 2
            #rands = torch.normal(x, std=0.3)
            #x = torch.clip(rands, 0, 1)
            #x = img# / 2
            #x = x + x * np.random.randn()#np.random.normal(size=x.shape, scale=5)
            #x = x + np.random.randn()
            #img[:,:,:] = x
            ds.images[i] = ds.images[i] + np.random.normal(size=ds.images[i].shape, scale=0.5)
            ds_c.images[i] = ds_c.images[i] + np.random.normal(size=ds_c.images[i].shape, scale=2) + np.random.randn()

        class_names = np.unique(ds.labels)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(MockNet().to(device))
        model = cast(MockNet, model)

        model = reset_parameters(model)
        # Run the model with each substep of data
        # train on subset of train data
        custom_train_class(
            model,
            ds,
            device,
            epochs=30
        )

        method = "classification_multiclass"
        metrics = ["accuracy"]
        estimator = LossEstimator(method, metrics)
        #results = estimator.evaluate(model, ds, ds_c, ds.class_names, 2)
        ds_c_outputs, _ = custom_eval(model, ds_c, ds_c.class_names, False)
        ds_c_dict = outputs_to_nannyml("classification", ds_c_outputs, ds_c.class_names)

        ds_outputs, ds_labels = custom_eval(model, ds, ds.class_names, True)
        ds_dict = outputs_to_nannyml("classification", ds_outputs, ds.class_names, ds_labels)

        results = estimator.evaluate(ds_dict, ds_c_dict, class_names, 2)

        ds_acc = results["Reference_accuracy"]
        ds_c_acc = results["Op_Predicted_accuracy"]

        assert ds_c_acc < 0.95*ds_acc
    
    def test_predicts_accuracy(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True
        np.random.seed(0)

        ds = LargeMockDataset()
        ds_c = LargeMockDataset()
        
        for i, img in enumerate(ds_c.images):
            ds.images[i] = ds.images[i] + np.random.normal(size=ds.images[i].shape, scale=0.5)
            ds_c.images[i] = ds_c.images[i] + np.random.normal(size=ds_c.images[i].shape, scale=2) + np.random.randn()

        class_names = np.unique(ds.labels)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(MockNet().to(device))
        model = cast(MockNet, model)


        model = reset_parameters(model)
        # Run the model with each substep of data
        # train on subset of train data

        custom_train_class(
            model,
            ds,
            device,
            epochs=30
        )

        method = "classification_multiclass"
        metrics = ["accuracy"]
        estimator = LossEstimator(method, metrics)
        #results = estimator.evaluate(model, ds, ds_c, ds.class_names, 2)
        ds_c_outputs, _ = custom_eval(model, ds_c, ds_c.class_names, False)
        ds_c_dict = outputs_to_nannyml("classification", ds_c_outputs, ds_c.class_names)

        ds_outputs, ds_labels = custom_eval(model, ds, ds.class_names, True)
        ds_dict = outputs_to_nannyml("classification", ds_outputs, ds.class_names, ds_labels)

        results = estimator.evaluate(ds_dict, ds_c_dict, class_names, 2)

        #ds_acc = results["Reference_Metric"]
        pred_ds_c_acc = results["Op_Predicted_accuracy"]
        true_ds_c_acc = get_accuracy(model, ds_c, ds_c.class_names)["Accuracy"].float()

        percent_diff = np.abs(pred_ds_c_acc - true_ds_c_acc)* 100
        assert percent_diff < 30 # Should see if we can get this bound lower


    
    def inactive_test_regress_degrades_on_corrupted_dataset(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True
        np.random.seed(0)

        ds = LargeMockDataset()
        ds_c = LargeMockDataset()
        
        for i, img in enumerate(ds_c.images):
            #img[0,:,:] -= 1.5
            #img[1,:,:] += 1.5
            #img[:,:,:] + 
            #x = img / 2
            #rands = torch.normal(x, std=0.3)
            #x = torch.clip(rands, 0, 1)
            #x = img# / 2
            #x = x + x * np.random.randn()#np.random.normal(size=x.shape, scale=5)
            #x = x + np.random.randn()
            #img[:,:,:] = x
            ds.images[i] = ds.images[i] + np.random.normal(size=ds.images[i].shape, scale=0.5)
            ds_c.images[i] = ds_c.images[i] + np.random.normal(size=ds_c.images[i].shape, scale=2) + np.random.randn()



        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(RegressNet().to(device))
        model = cast(RegressNet, model)


        model = reset_parameters(model)
        # Run the model with each substep of data
        # train on subset of train data

        custom_train_regress(
            model,
            ds,
            device,
            epochs=30
        )

        method = "regression"
        estimator = LossEstimator(method, ["rmse"])
        results = estimator.evaluate(model, ds, ds_c, ds.class_names, 2)

        ds_err = results["Reference_rmse"]
        ds_c_err = results["Op_Predicted_rmse"]

        assert ds_err < 0.95*ds_c_err