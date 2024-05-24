import os
from typing import Any, Dict, Tuple, Union

import drift_projnorm as dd
import numpy as np
import pytest
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from tests.utils.data import DamlDataset

np.random.seed(0)
torch.manual_seed(0)


class TestDrift:
    def test_projnorm(self):
        weights_1 = [np.array([[1, 1], [1, 1]]), np.array([[2, 1], [1, 1]])]

        weights_2 = [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, -1]])]

        desired_output = np.sqrt(1 + 1 + 1 + 1 + 4 + 1 + 1 + 4)

        drift_metric = dd.DriftDetector()

        computed_output = drift_metric._compute_projnorm(weights_1, weights_2)

        assert desired_output == computed_output

    def test_forcedlabeldataset(self, mnist):
        model = ConstantNet()

        train_ds = DamlDataset(*mnist(50, "train", np.float32, "channels_first"))

        forced_label_ds = dd.DatasetForcedLabels(model, train_ds)

        for idx in range(len(forced_label_ds)):
            img, label = forced_label_ds[idx]
            assert label == 2

    @pytest.mark.skip
    def test_underspec(self, mnist):
        """Unit testing of drift detection

        TBD - run takes too long
        """
        model = Net()

        train_ds = DamlDataset(*mnist(1000, "train", np.float32, "channels_first"))

        drift_metric = dd.DriftDetector()

        drift_metric.set_training_func(custom_train)
        drift_metric.set_eval_func(custom_eval)

        drift_metric.set_train_eval_params()

        underspec = drift_metric._is_underspecified(
            model=model,
            train_ds=train_ds,
            underspec_runs=3,
        )
        assert underspec

    # @pytest.mark.skip
    def test_drift(self, mnist):
        """Unit testing of drift detection"""

        model = Net()

        train_ds = DamlDataset(*mnist(1000, "train", np.float32, "channels_first"))
        test_ds = DamlDataset(*mnist(100, "test", np.float32, "channels_first"))

        test_ds_1 = torch.utils.data.Subset(test_ds, range(30))  # test_ds[:30]
        test_ds_2 = torch.utils.data.Subset(test_ds, range(30, 60))  # test_ds[30:60]
        test_ds_3 = torch.utils.data.Subset(test_ds, range(60, 100))  # test_ds[60:]

        drift_metric = dd.DriftDetector()

        drift_metric.set_training_func(custom_train)
        drift_metric.set_eval_func(custom_eval)

        drift_metric.set_train_eval_params()

        labeled_test_datasets = [test_ds_1, test_ds_2]
        unlabeled_test_datasets = [test_ds_3]

        pred_accuracies, _ = drift_metric.run(model, train_ds, labeled_test_datasets, unlabeled_test_datasets)

        pnorm, true_accuracy = drift_metric.get_projnorm_accuracy(
            model=model, train_ds=train_ds, test_ds=test_ds_3, labeled_test=True
        )
        # Check that we have a good estimate of the accuracy
        percent_error = 100 * abs(true_accuracy - pred_accuracies[0]) / true_accuracy

        assert percent_error < 4

    @pytest.mark.skip
    def test_drift_different_data(self, mnist):
        """Unit testing of drift detection

        TBD - currently not working
        """

        model = Net()

        train_ds = DamlDataset(*mnist(1000, "train", np.float32, "channels_first"))
        test_ds = DamlDataset(*mnist(200, "test", np.float32, "channels_first"))

        test_ds_1 = torch.utils.data.Subset(test_ds, range(100, 130))  # test_ds[:30]
        test_ds_2 = torch.utils.data.Subset(test_ds, range(130, 160))  # test_ds[30:60]
        test_ds_3 = torch.utils.data.Subset(test_ds, range(160, 200))  # test_ds[60:]

        drift_metric = dd.DriftDetector()

        drift_metric.set_training_func(custom_train)
        drift_metric.set_eval_func(custom_eval)

        drift_metric.set_train_eval_params()

        labeled_test_datasets = [test_ds_1, test_ds_2]
        unlabeled_test_datasets = [test_ds_3]

        pred_accuracies, _ = drift_metric.run(model, train_ds, labeled_test_datasets, unlabeled_test_datasets)

        pnorm, true_accuracy = drift_metric.get_projnorm_accuracy(
            model=model, train_ds=train_ds, test_ds=test_ds_3, labeled_test=True
        )
        # Check that we have a good estimate of the accuracy
        percent_error = 100 * abs(true_accuracy - pred_accuracies[0]) / true_accuracy

        assert percent_error < 10

    def test_drift_many_datasets(self, mnist):
        """Unit testing of drift detection

        TBD - currently not working
        """

        model = Net()

        train_ds = DamlDataset(*mnist(1000, "train", np.float32, "channels_first"))
        test_ds = DamlDataset(*mnist(2500, "test", np.float32, "channels_first"))

        test_ds_1 = torch.utils.data.Subset(test_ds, range(300))
        test_ds_2 = torch.utils.data.Subset(test_ds, range(300, 600))
        test_ds_3 = torch.utils.data.Subset(test_ds, range(600, 1000))
        test_ds_4 = torch.utils.data.Subset(test_ds, range(1000, 1300))
        test_ds_5 = torch.utils.data.Subset(test_ds, range(1300, 1600))
        test_ds_6 = torch.utils.data.Subset(test_ds, range(1600, 2000))

        drift_metric = dd.DriftDetector()

        drift_metric.set_training_func(custom_train)
        drift_metric.set_eval_func(custom_eval)

        drift_metric.set_train_eval_params()

        labeled_test_datasets = [test_ds_1, test_ds_2, test_ds_3, test_ds_4]
        # labeled_test_datasets = [test_ds_1, test_ds_2]  # , test_ds_4]
        # test_ds_3 is a problem, apparently
        # unlabeled_test_datasets = [test_ds_5, test_ds_6]
        unlabeled_test_datasets = [test_ds_5]

        pred_accuracies, _ = drift_metric.run(model, train_ds, labeled_test_datasets, unlabeled_test_datasets)

        pnorm, true_accuracy_5 = drift_metric.get_projnorm_accuracy(
            model=model, train_ds=train_ds, test_ds=test_ds_5, labeled_test=True
        )
        pnorm, true_accuracy_6 = drift_metric.get_projnorm_accuracy(
            model=model, train_ds=train_ds, test_ds=test_ds_6, labeled_test=True
        )
        # Check that we have a good estimate of the accuracy
        percent_error_5 = 100 * abs(true_accuracy_5 - pred_accuracies[0]) / true_accuracy_5
        percent_error_6 = 100 * abs(true_accuracy_6 - pred_accuracies[1]) / true_accuracy_6

        assert percent_error_5 < 4
        assert percent_error_6 < 4

    def test_corrupted(self, mnist):
        # model = Net()
        model = ResNet18(num_classes=10, seed=1)

        trainset = load_cifar10_image(
            corruption_type="clean",
            clean_cifar_path="./data",
            corruption_cifar_path="./data/cifar/CIFAR-10-C",
            corruption_severity=0,
            num_samples=1000,
            datatype="train",
        )

        random_seeds = [1, 2, 3]
        valset_iid = load_cifar10_image(
            corruption_type="clean",
            clean_cifar_path="./data",
            corruption_cifar_path="./data/cifar/CIFAR-10-C",
            corruption_severity=0,
            datatype="test",
            num_samples=1000,
            seed=random_seeds[0],
        )
        valset_iid2 = load_cifar10_image(
            corruption_type="clean",
            clean_cifar_path="./data",
            corruption_cifar_path="./data/cifar/CIFAR-10-C",
            corruption_severity=0,
            datatype="test",
            num_samples=1000,
            seed=random_seeds[1],
        )

        valset_ood = load_cifar10_image(
            corruption_type="snow",
            clean_cifar_path="./data",
            corruption_cifar_path="./data/cifar/CIFAR-10-C",
            corruption_severity=5,
            datatype="test",
            seed=random_seeds[2],
        )

        # train_ds = DamlDataset(*mnist(10000, "train", np.float32, "channels_first"))

        # id_test = DamlDataset(*mnist(2000, "test", np.float32, "channels_first"))
        # id_test_1 = torch.utils.data.Subset(id_test, range(0, 1000))
        # id_test_2 = torch.utils.data.Subset(id_test, range(1000, 2000))
        labeled_test_datasets = [valset_iid, valset_iid2]

        ood_test = TFDataset("mnist_corrupted/translate", "test", 1000)
        unlabeled_test_datasets = [valset_ood]

        drift_metric = dd.DriftDetector()
        drift_metric.set_training_func(custom_train)
        drift_metric.set_eval_func(custom_eval)
        drift_metric.set_train_eval_params()

        pred_accuracies, _ = drift_metric.run(model, trainset, labeled_test_datasets, unlabeled_test_datasets)  # type: ignore

        pnorm, true_accuracy = drift_metric.get_projnorm_accuracy(
            model=model, train_ds=trainset, test_ds=ood_test, labeled_test=True
        )

        percent_error = 100 * abs(true_accuracy - pred_accuracies[0]) / true_accuracy
        print(pred_accuracies)
        print(true_accuracy)
        assert percent_error < 10


class TFDataset(Dataset):
    """Holds the arrays of images and labels"""

    def __init__(self, dataset_name, split, length) -> None:
        # ds = tfds.load("mnist_corrupted/translate", split="train", with_info=True)
        # c, d = ds

        ds_elements, _ = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
        )  # type: ignore

        self._images: Any = np.array([i["image"].numpy() for i in list(ds_elements.take(length))], dtype=np.float32)  # type: ignore
        # Convert to channels_first format
        # self._images = self._images[:, np.newaxis]
        self._images = self._images.transpose((0, 3, 1, 2))
        self._labels = np.array([i["label"].numpy() for i in list(ds_elements.take(length))], dtype=np.uint8)  # type: ignore

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index) -> Union[Any, Tuple]:
        image: Any = self._images[index]

        # Return image if no other attributes
        if self._labels is None:
            return image

        labels: Any = self._labels[index]
        # Return image and label for image classification

        # Return images, labels, boxes for object detection
        return image, labels

    @property
    def images(self) -> Any:
        return self._images

    @property
    def labels(self) -> Any:
        if self._labels is None:
            return np.array([])
        else:
            return self._labels


def custom_eval(model: nn.Module, dataloader) -> Dict[str, float]:
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

    Returns
    -------
    float
        The calculated performance of the model
    """
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    # Set model layers into evaluation mode
    model.eval()
    # Tell PyTorch to not track gradients, greatly speeds up processing
    result: float = 0.0
    batch_accuracies = np.zeros(len(dataloader), dtype=float)
    for batch_idx, data in enumerate(dataloader):
        X, y = data
        with torch.no_grad():
            preds = model(X)
            batch_accuracy = metric(preds, y)
            batch_accuracies[batch_idx] = batch_accuracy
    result = float(np.mean(batch_accuracies))
    return {"Accuracy": result}


def custom_train(model: nn.Module, dl: DataLoader):
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
    # Defined only for this testing scenario
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 20
    for epoch in range(epochs):
        for X, y in dl:
            # Zero out gradients
            optimizer.zero_grad()
            # Forward Propagation
            outputs = model(X)
            # Back prop
            loss = criterion(outputs, y)
            loss.backward()
            # Update optimizer
            optimizer.step()


class MockNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, x):
        return x

    def forward(self, x):
        pass


class ConstantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, x):
        return x

    def forward(self, x):
        return torch.tensor([[0.0, 0.0, 1.0, 0.0]])


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(6400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_cls_dataset() -> Tuple[DamlDataset, DamlDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))

    train_ds = DamlDataset(images, labels)
    test_ds = DamlDataset(images, labels)

    return train_ds, test_ds


def load_od_dataset() -> Tuple[DamlDataset, DamlDataset]:
    images = np.ones(shape=(1, 32, 32, 3))
    labels = np.ones(shape=(1, 1))
    boxes = np.ones(shape=(1, 1, 4))

    train_ds = DamlDataset(images, labels, boxes)
    test_ds = DamlDataset(images, labels, boxes)

    return train_ds, test_ds


def eval_100(model: nn.Module, dl: DataLoader) -> Dict[str, float]:
    """Eval should always return a float, and error if not"""
    return {"eval": 1.0}


def load_cifar10_image(
    corruption_type,
    clean_cifar_path,
    corruption_cifar_path,
    corruption_severity=0,
    datatype="test",
    num_samples=50000,
    seed=1,
):
    """
    Returns:
        pytorch dataset object
    """
    assert datatype == "test" or datatype == "train"
    training_flag = datatype == "train"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset = datasets.CIFAR10(clean_cifar_path, train=training_flag, transform=transform, download=True)

    if corruption_severity > 0:
        assert not training_flag
        path_images = os.path.join(corruption_cifar_path, corruption_type + ".npy")
        path_labels = os.path.join(corruption_cifar_path, "labels.npy")

        dataset.data = np.load(path_images)[(corruption_severity - 1) * 10000 : corruption_severity * 10000]
        dataset.targets = list(np.load(path_labels)[(corruption_severity - 1) * 10000 : corruption_severity * 10000])
        dataset.targets = [int(item) for item in dataset.targets]

    # randomly permute data
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    number_samples = dataset.data.shape[0]
    index_permute = torch.randperm(number_samples)
    dataset.data = dataset.data[index_permute]
    dataset.targets = (np.array([int(item) for item in dataset.targets])[index_permute]).tolist()

    # randomly subsample data
    if datatype == "train" and num_samples < 50000:
        indices = torch.randperm(50000)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print("number of training data: ", len(dataset))
    if datatype == "test" and num_samples < 10000:
        indices = torch.randperm(10000)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print("number of test data: ", len(dataset))

    return dataset


def ResNet18(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(512, num_classes)
    return resnet18
