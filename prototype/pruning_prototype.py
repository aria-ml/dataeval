import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ops import Permute
from torch.utils.data import DataLoader, RandomSampler, Subset, default_collate
from torchmetrics.classification import MulticlassAveragePrecision
from torchvision.transforms import Compose, PILToTensor, ToTensor
from tqdm import tqdm

# Dataset and model
from dataeval.utils.data import Embeddings
from dataeval.utils.data.datasets import CIFAR10
# from dataeval.utils.data.selections import ClassFilter, Limit, Shuffle

from pruning_prototype_helpers import ResierResNet18,evaluate,fine_tune,freeze_all_but,unfreeze_all
from pruning_utils import KNNSorter,ClusterSorter,KMeansSorter,RandomSorter

# Set default torch device for notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

if __name__=="__main__":

    bs = 256
    embedding_dim = 16
    transforms = Compose([ToTensor(), lambda x: torch.permute(x, [1,0,2]).to(device)])
    d_trn = CIFAR10(
        "./data",
        image_set="train",
        download=True,
        verbose=False,
        transforms= transforms,
    )
    loader = DataLoader(d_trn, batch_size=bs)
    num_classes = len(d_trn.index2label.keys())

    d_tst = CIFAR10(
        "./data",
        image_set="test",
        download=True,
        verbose=False,
        transforms=transforms
    )
    tst_loader = DataLoader(d_tst, batch_size=bs)

    # load pretrained model with additional embedding and classification layer
    resnet = ResierResNet18(embedding_size=embedding_dim, num_classes=num_classes)
    # fine tune last two layers
    resnet = freeze_all_but(resnet, ["fc", "fc2class"])
    resnet = fine_tune(resnet, loader, num_epochs=1, lr=0.001)
    # fine tune everything
    resnet = unfreeze_all(resnet)
    resnet = fine_tune(resnet, loader, num_epochs=5, lr=0.0001)

    # embedded dataset
    embeddings = Embeddings(dataset=d_trn, batch_size=bs, model=resnet).to_tensor().cpu().numpy()
    # normalization is important for the ClusterSorter
    embeddings /= np.max(np.linalg.norm(embeddings, axis=1))
    N0 = embeddings.shape[0]

    # pruning here
    k_knn = 100
    nclst = 1000
    # sorted easiest/most-prototypical first
    sorters = [
        KNNSorter(embeddings, k=k_knn),
        KMeansSorter(embeddings,num_clusters=nclst),
        ClusterSorter(embeddings, num_clusters=nclst),
        RandomSorter(embeddings)
    ]

    # not necessary for testing functionality; more for performance
    nums_samp = [5000, 10000, 25000, 50000]
    metric = {}
    for srt in sorters:
        metric[srt.name] = []
        for ns in nums_samp:
            # we could wrap this for differnet policies, but in our experiments
            # keep_hard (indexing -ns:) performs better
            pruned = Subset(d_trn, srt.srt_inds[-ns:])
            pruned_loader = DataLoader(pruned, batch_size=bs)
            model = ResierResNet18(embedding_size=embedding_dim, num_classes=num_classes)
            model = fine_tune(model, pruned_loader, num_epochs=15, lr=0.001)
            print(f"Trained model on prioritized dataset ({srt.name}:{ns} samples)")
            metric[srt.name].append(evaluate(tst_loader, model))
            print(f"AP ({srt.name}:{ns} samples): {metric[srt.name][-1]}")

    with open(f"tmp/pruning_test_{embedding_dim}D.pkl", 'wb') as f:
            pickle.dump(metric, f)
