from dataeval.utils.torch.models import ResNet18
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torchmetrics.classification import MulticlassAveragePrecision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResierResNet18(ResNet18):
    """
    Wrapper for the ResNet18 class to add a classification layer and to separate
    embedding and the classification forward pass.
    """
    def __init__(self, embedding_size=16, num_classes=-1):
        super().__init__(embedding_size=embedding_size)
        # embed to 'embedding_size' then to number of classes
        self.model.fc2class = nn.Linear(embedding_size, num_classes)

    def forward(self, x, embed_only=True):
        # default to embedding only to use with Embedding class
        return self.model(x) if embed_only else self.model.fc2class(self.model(x))

def fine_tune(model, loader, num_epochs=5, lr=0.001):
    """
    Quick and dirty training function
    """
    # Defined only for this testing scenario
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = torch.tensor([0.0], requires_grad=True)

    for _ in range(num_epochs):
        for i,batch in tqdm(enumerate(loader), total=np.ceil(len(loader) / loader.batch_size)):
            ims,labels,md = batch
            # Zero out gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(ims.float(), embed_only=False)
            # Compute loss
            loss = criterion(outputs, labels)
            # Back prop
            loss.backward()
            # Update weights/parameters
            optimizer.step()
    return model

def freeze_all_but(model, not_freeze):
    """
    freeze all layers but those named in 'not_freeze'
    """
    for nm, param in model.named_parameters():
        if all([n not in nm for n in not_freeze]):
            param.requires_grad = False
        else:
            print(f"{nm} not frozen")
    return model

def unfreeze_all(model):
    """
    unfreeze all model layers
    """
    for param in model.parameters():
        param.requires_grad = True
    return model

def evaluate(loader, model):
    """
    utility for developing prototype
    """
    metric = MulticlassAveragePrecision(num_classes=10)
    model.eval()
    with torch.no_grad():
        for im,target,md in loader:
            pred = model(im, embed_only=False)
            metric(pred, target.argmax(axis=1))
    return metric.compute()