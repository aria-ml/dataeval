
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fmow_utils import extrinsic_factors_fmow, get_fmow_boxes
from intrinsic_factors import intrinsic_factors_xywh
from torch.utils.data import Dataset, DataLoader, default_collate
import torch
from matplotlib.patches import Rectangle
from torchmetrics import MetricCollection

from dataeval._internal.metrics.metadata import Balance, BalanceClasswise, Diversity, DiversityClasswise

def balance_classwise_fig(mi, metric, class_names):
    f, ax = plt.subplots(figsize=(12, 8))
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns_plot = sns.heatmap(
        mi,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": "Normalized Mutual Information"},
        xticklabels=metric.names[1:],
        yticklabels=class_names,
        annot=True,
    )
    plt.xlabel("Class")
    plt.tight_layout(pad=0)


def balance_fig(mi, metric):
    fig, ax = plt.subplots(figsize=(12, 8))
    # mask out lower triangular portion
    mask = np.zeros_like(mi, dtype=np.bool_)
    mask[np.tril_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = True
    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(
        np.minimum(mi[:, 1:], 1),
        mask=mask[:, 1:],
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": "Normalized Mutual Information"},
        xticklabels=metric.names[1:],
        yticklabels=metric.names[:-1],
        annot=True,
    )
    # highlight correlation with class
    ax.add_patch(Rectangle((0, 0), mi.shape[0], 1, fill=False, edgecolor="w", lw=6))
    plt.tight_layout(pad=0)
    return fig

def diversity_fig(div, metric):
    # plt_df = pd.DataFrame({"Simpson": div_simpson, "Shannon": div_shannon})
    fig= plt.figure(figsize=(10,5))
    plt.bar(x=np.arange(len(div)), height=div, tick_label=metric.names)
    plt.tick_params(labelrotation=90)
    plt.ylabel("Diversity")
    # plt.gca().set_xticklabels(metric.names)
    plt.tight_layout(pad=0)
    return fig

def diversity_classwise_fig(div, metric, class_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns_plot = sns.heatmap(
        div,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Diversity Index"},
        xticklabels=[n for n in metric.names if n != "class_label"],
        yticklabels=class_names,
        annot=True,
    )
    plt.xlabel("Factors")
    plt.ylabel("Class Label")
    plt.title("")
    plt.tight_layout(pad=0)
    return fig

class MyMaiteDataset(Dataset):
    def __init__(self, df):
        self.metadata = []
        self.class_labels = torch.tensor(df["class"])
        df.drop("class", axis=1, inplace=True)
        for _,row in df.iterrows():
            self.metadata.append({k: row[k] for k in df.columns})
    def __getitem__(self, idx):
        # placeholder for images
        return torch.empty(0),self.class_labels[idx],self.metadata[idx]
    def __len__(self):
        return len(self.metadata)

def maite_collate(batch):
    ims = default_collate([im for im,_,_ in batch])
    lbs = default_collate([lab for _,lab,_ in batch])
    md = [md for _,_,md in batch]
    return ims,lbs,md

if __name__=="__main__":

    demo_classes = [
        "airport",
        "border_checkpoint",
        "dam",
        "factory_or_powerplant",
        "hospital",
        "military_facility",
        "nuclear_powerplant",
        "oil_or_gas_facility",
        "place_of_worship",
        "port",
        "prison",
        "stadium",
        "electric_substation",
        "road_bridge",
    ]

    split_name = "dev"
    tvt_splits = ["train", "val"]
    dfs = []
    for split in tvt_splits:
        dfs.append(pd.read_csv(f"prototype/splits/{split_name}_{split}.csv"))
    df = pd.concat(dfs)

    class_str = np.unique(df["class"])

    ext_factors, _ = extrinsic_factors_fmow(df)
    # assume class label is in the dataframe
    _, ext_factors["class"] = np.unique(df["class"], return_inverse=True)
    # back to dataframe because I originally wrote the dataset for dataframe
    df = pd.DataFrame(ext_factors)

    for idx, row in df.iterrows():
        d = {k: row[k] for k in df.columns}
    dataset = MyMaiteDataset(df)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=maite_collate)

    # Individual metrics
    balance = Balance()
    balance_classwise = BalanceClasswise()
    diversity = Diversity(metric="simpson")
    diversity_classwise = DiversityClasswise(metric="shannon")

    # MetricCollection: the compute group hopefully only aggregates data once since each metric
    # has a common state.
    bias = MetricCollection([Balance(), BalanceClasswise(), Diversity(metric="simpson"), DiversityClasswise(metric="simpson")],
                compute_groups=[["Balance", "BalanceClasswise", "Diversity", "DiversityClasswise"]])

    for im, lab, md in loader:
        bias.update(lab, md)

    metrics = bias.compute()

    fig = balance_classwise_fig(metrics["BalanceClasswise"], bias.BalanceClasswise, class_str)
    plt.savefig("prototype/figs/balance_classwise.png")

    fig = balance_fig(metrics["Balance"], bias.Balance)
    plt.savefig("prototype/figs/balance.png")

    fig = diversity_fig(metrics["Diversity"], bias.Diversity)
    plt.savefig("prototype/figs/diversity.png")

    fig = diversity_classwise_fig(metrics["DiversityClasswise"], bias.DiversityClasswise, class_str)
    plt.savefig("prototype/figs/diversity_classwise.png")
