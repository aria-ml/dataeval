
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (ConcatDataset, DataLoader, Subset,
                              default_collate)
from torchvision.transforms import Compose, ToTensor

# Dataset and model
from dataeval.utils.data import Embeddings
from dataeval.utils.data.datasets import CIFAR10
# from dataeval.utils.data.selections import ClassFilter, Limit, Shuffle

# Set default torch device for notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

from pruning_prototype_helpers import ResierResNet18,evaluate,fine_tune
from pruning_utils import prioritize


if __name__ == "__main__":

    bs = 256
    embedding_dim = 16
    transforms = Compose([ToTensor(), lambda x: torch.permute(x, [1,0,2]).to(device)])
    d_trn_0 = CIFAR10(
        "./data",
        image_set="train",
        download=True,
        verbose=False,
        transforms= transforms,
    )
    n_trn = 35000
    d_trn, d_cnd = torch.utils.data.random_split(d_trn_0, [0.7, 0.3], generator=torch.Generator(device="cuda"))

    trn_loader = DataLoader(d_trn, batch_size=bs)
    cnd_loader = DataLoader(d_cnd, batch_size=bs)
    num_classes = len(d_trn_0.index2label.keys())

    d_tst = CIFAR10(
        "./data",
        image_set="test",
        download=True,
        verbose=False,
        transforms=transforms
    )
    tst_loader = DataLoader(d_tst, batch_size=bs)

    # train model to embed
    resnet = ResierResNet18(embedding_size=embedding_dim, num_classes=num_classes)
    # fine tune
    resnet = fine_tune(resnet, trn_loader, num_epochs=10, lr=0.001)


    # embedded dataset
    emb_trn = Embeddings(dataset=d_trn, batch_size=bs, model=resnet).to_tensor().cpu().numpy()
    emb_trn /= np.max(np.linalg.norm(emb_trn, axis=1))
    # embedded candidate data
    emb_cnd = Embeddings(dataset=d_cnd, batch_size=bs, model=resnet).to_tensor().cpu().numpy()
    emb_cnd /= np.max(np.linalg.norm(emb_cnd, axis=1))

    # prioritized candidate sample indices
    knn_prio_inds = prioritize(emb_trn, emb_cnd, method = 'knn', strategy = 'keep_hard', k = 100)

    # add 10% of candidate data
    n_prio = int(0.1*len(d_cnd))

    # In principle the candidate data are unlabeled, so we may not be able to
    #   immediately select a dataset as we do here
    comb_train_ds = ConcatDataset([d_trn, Subset(d_cnd, knn_prio_inds[:n_prio])]) # type: ignore
    comb_loader = DataLoader(comb_train_ds, batch_size=bs)
    # train model on combined dataset
    model = ResierResNet18(embedding_size=embedding_dim, num_classes=num_classes)
    model = fine_tune(model, comb_loader, num_epochs=10, lr=0.001)
    ap = evaluate(tst_loader, model)



