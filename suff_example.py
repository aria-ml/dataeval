from typing import Dict, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset

import daml._internal.metrics.sufficiency as dms
from daml.metrics import Sufficiency

#torch._dynamo.config.suppress_errors = True

np.random.seed(0)
np.set_printoptions(formatter={"float": lambda x: f"{x:0.4f}"})
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download the mnist dataset and preview the images
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=to_tensor)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=to_tensor)

fig = plt.figure(figsize=(8, 3))

for lbl in range(10):
    i = (train_ds.targets == lbl).nonzero()[0][0]
    img = train_ds.data[i]
    ax = fig.add_subplot(2, 5, lbl + 1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(img, cmap="gray_r")

    # Define our network architecture
class Net(nn.Module):
    def __init__(self):
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


# Compile the model
model = torch.compile(Net().to(device))

# Type cast the model back to Net as torch.compile returns a Unknown
# Nothing internally changes from the cast; we are simply signaling the type
model = cast(Net, model)

def custom_train(model: nn.Module, dataset: Dataset, indices: Sequence[int]):
    # Defined only for this testing scenario
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 10

    # Define the dataloader for training
    dataloader = DataLoader(Subset(dataset, indices), batch_size=16)

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


def custom_eval(model: nn.Module, dataset: Dataset) -> Dict[str, float]:
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    result = 0

    # Set model layers into evaluation mode
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16)
    # Tell PyTorch to not track gradients, greatly speeds up processing
    with torch.no_grad():
        for batch in dataloader:
            # Load data/images to device
            X = torch.Tensor(batch[0]).to(device)
            # Load targets/labels to device
            y = torch.Tensor(batch[1]).to(device)
            preds = model(X)
            metric.update(preds, y)
        result = metric.compute()
    return {"Accuracy": result}

# Instantiate sufficiency metric
suff = Sufficiency(
    model=model,
    train_ds=train_ds,
    test_ds=test_ds,
    train_fn=custom_train,
    eval_fn=custom_eval,
    runs=5,
    substeps=10,
)

# Train & test model
output = suff.evaluate()

# Print out sufficiency output in a table format
from tabulate import tabulate

print(tabulate(output, headers=list(output.keys()), tablefmt="pretty"))

# Plot the output using the convenience function
_ = Sufficiency.plot(output)

desired_accuracies = [0.1, 0.5, 0.9, 0.99]
suff_datas = dms.data_to_produce_accuracy(desired_accuracies, output)
print(suff_datas)