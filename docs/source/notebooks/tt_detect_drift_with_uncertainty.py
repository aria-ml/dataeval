# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Detect drift with prediction uncertainty
#
# Learn how to monitor a deployed image classifier for distribution shift with the model's own prediction uncertainty,
# and how to check what that monitor can and cannot see.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Monitoring](../getting-started/roles/ML_Lifecycle.md#monitoring)
#
# Relevant personas: [ML Engineer](../getting-started/roles/ml_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)
#
# ## What you will do
#
# - Train a vehicle-family classifier on MilitaryVehicles images, and hold one family out of training
# - Turn the classifier into a label-free uncertainty signal with {class}`.UncertaintyExtractor`
# - Monitor batches of images with the KS test of {class}`.DriftUnivariate`, and measure how often it raises a false
#   alarm
# - Degrade the images by known amounts, and compare the alarms with the accuracy that you would not see in deployment
# - Test what the monitor misses: a vehicle family that the classifier never saw, a change in the mix of families, and
#   an overconfident classifier
#
# ## What you will learn
#
# - How prediction entropy acts as a drift signal when you have no labels
# - How to measure the false alarm rate of a monitor on clean batches before you rely on it
# - That the monitor flags moderate degradation reliably, that it is not more sensitive than a simple feature check,
#   and that an alarm does not tell you how much accuracy was lost
# - That an overconfident classifier weakens the monitor
#
# ## What you will need
#
# - Knowledge of Python
# - Basic understanding of PyTorch and image classification

# %% [markdown]
# ## Introduction
#
# Most drift detectors compare the features of incoming data against a reference set (see the
# [monitoring tutorial](./tt_monitor_shift.py)). A deployed classifier gives you another signal. As operational data
# moves into regions that the model was not trained on, its predictions can become less confident.
#
# You can summarize the confidence of a classifier as the {term}`Shannon entropy<Shannon Entropy>` of its predicted
# class probabilities. Low entropy means a confident, peaked prediction. High entropy means the model spreads its
# confidence across classes. If the distribution of entropy shifts on operational data, the model is behaving
# differently, and you learn that without any labels. See the
# [uncertainty-based drift detection](../concepts/DistributionShift.md#uncertainty-based-drift-detection) concept page
# for the theory and trade-offs.
#
# In this tutorial you are the test engineer for a classifier that names the family of a military vehicle: tank, BMP,
# BTR, self-propelled artillery, MT-LB or BMD. It will be fielded where nobody labels the images that it sees. You want
# an alarm when it starts to struggle, so you build a monitor from its own uncertainty and then check, with images that
# you degrade by known amounts, what the monitor really tells you.

# %% [markdown]
# ## Setup
#
# Import the libraries that you need.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets torchvision
except Exception:
    pass

# %%
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
import torch
from maite_datasets.image_classification import MilitaryVehicles
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import radius_neighbors_graph
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.config import set_batch_size, set_device
from dataeval.extractors import ClasswiseUncertaintyExtractor, TorchExtractor, UncertaintyExtractor
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Duplicates
from dataeval.shift import DriftUnivariate

# Use the GPU if one is available, otherwise the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

# Set the default batch size used when loading and encoding images
set_batch_size(64)

# Render the plotly figures below with the plotly JS embedded in the notebook output
pio.renderers.default = "notebook"

# %% [markdown]
# ## Load the dataset
#
# Load the train and test splits of MilitaryVehicles with `MilitaryVehicles` from `maite_datasets`. The dataset ships
# with a hierarchy that groups its 24 vehicle types into families. You classify the family of a vehicle, and you hold
# one family, the one that the dataset labels Air Defense, out of everything that the classifier trains and calibrates
# on. Its images play the part of a family that the classifier never saw.

# %%
train = MilitaryVehicles("./data", image_set="train", download=True)
test = MilitaryVehicles("./data", image_set="test", download=True)

land = train.hierarchy["vehicle"]["military"]["land vehicle"]
family_of_class = {train.label2index[name]: family for family, names in land.items() for name in names}
family_names = sorted(set(family_of_class.values()) - {"Air Defense"})

labels = np.array([int(np.argmax(target)) for dataset in (train, test) for _, target, _ in dataset])
unseen = np.array([family_of_class[label] == "Air Defense" for label in labels])
known = np.flatnonzero(~unseen)
family = np.array([
    family_names.index(family_of_class[label]) if not hole else -1 for label, hole in zip(labels, unseen, strict=True)
])
print(f"{len(known):,} images in {len(family_names)} families, and {unseen.sum():,} images of the unseen family")


def photo(index):
    """The image at an index that counts over the train split and then the test split."""
    return np.asarray(train[index][0] if index < len(train) else test[index - len(train)][0])


# %% [markdown]
# ## Embed the images
#
# The classifier is a pretrained ResNet18 whose weights stay frozen, with a small linear head on top that you train. A
# {class}`.TorchExtractor` runs the ResNet18 up to its `avgpool` layer, and {class}`.Embeddings` turns every image into
# a 512-dimensional embedding. Because the backbone is frozen, you embed each image once and apply the head to the
# embeddings.

# %%
weights = ResNet18_Weights.DEFAULT
backbone = TorchExtractor(
    resnet18(weights=weights, progress=False), transforms=weights.transforms(), layer_name="avgpool"
)
embeddings = np.vstack([np.asarray(Embeddings(dataset, extractor=backbone)) for dataset in (train, test)])
print(f"embeddings: {embeddings.shape}")

# %% [markdown]
# ## Keep repeated images on one side
#
# Some images in the dataset repeat another image: copies, crops and resaved versions. A model looks better on a repeat
# of an image that it trained on, and a batch that holds repeats of the reference is not a fresh sample. So you sort the
# images into **families** of repeats and keep every family in one set. The cell below finds the repeats with
# {class}`.Duplicates` and with the similarity of the embeddings. [Assess an unlabeled data
# space](./tt_assess_data_space.py) and [Clean a dataset](./tt_clean_dataset.py) show repeats in this dataset in more
# detail.

# %% tags=["hide-input"]
found = Duplicates(hash_radius=0).evaluate(train, test).data()
pairs = []
for items, splits in zip(found["item_indices"], found["dataset_indices"], strict=True):
    ids = [item + split * len(train) for item, split in zip(items, splits, strict=True)]
    pairs += [(ids[0], other) for other in ids[1:]]
# Embeddings at cosine similarity >= 0.95, which for embeddings of length 1 is a distance <= sqrt(2 - 2 * 0.95)
unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
links = radius_neighbors_graph(unit, radius=np.sqrt(2 - 2 * 0.95))
links = links + coo_matrix((np.ones(len(pairs)), tuple(zip(*pairs, strict=True))), shape=links.shape)
_, repeat_family = connected_components(links, directed=False)
print(f"{(np.bincount(repeat_family)[repeat_family] > 1).mean():.0%} of the images belong to a family of repeats")

# %% [markdown]
# ## Split the images into three sets
#
# Split the images of the known families, by family of repeats, into three sets with different jobs:
#
# | Set | Images | Job |
# | --- | --- | --- |
# | Train | 3,000 | Fit the classifier head |
# | Reference | 2,000 | Hold the entropy that the monitor treats as normal. The classifier never trained on these. |
# | Operational | 1,500 | Stand in for the images that arrive after deployment |
#
# Never take the reference from the images that the classifier trained on. A model is more confident on its own training
# images, so their entropy is not what it will produce on new images.

# %%
rng = np.random.default_rng(0)
train_rows, rest = next(
    GroupShuffleSplit(n_splits=1, train_size=3000, random_state=0).split(known, groups=repeat_family[known])
)
reference_rows, operational_rows = next(
    GroupShuffleSplit(n_splits=1, train_size=2000, random_state=0).split(known[rest], groups=repeat_family[known[rest]])
)
train_idx, reference_idx = known[train_rows], known[rest][reference_rows]
operational_idx = known[rest][operational_rows][:1500]
print(f"train: {len(train_idx)}  reference: {len(reference_idx)}  operational: {len(operational_idx)}")

# %% [markdown]
# ## Train the classifier
#
# Fit a logistic regression on the embeddings of the training images. The strong regularization (`C=0.05`) keeps the
# classifier from becoming overconfident, and a later section shows why that matters. Then copy its weights into a
# PyTorch linear layer so that DataEval can run it as a model.

# %%
classifier = LogisticRegression(C=0.05, max_iter=3000).fit(embeddings[train_idx], family[train_idx])


def to_torch(fitted):
    """Copy a fitted logistic regression into a PyTorch linear layer."""
    layer = torch.nn.Linear(fitted.coef_.shape[1], fitted.coef_.shape[0])
    layer.weight.data = torch.tensor(fitted.coef_, dtype=torch.float32)
    layer.bias.data = torch.tensor(fitted.intercept_, dtype=torch.float32)
    return layer


head = to_torch(classifier)
print(
    f"accuracy on the operational images: {classifier.score(embeddings[operational_idx], family[operational_idx]):.0%}"
)

# %% [markdown]
# ## Build the uncertainty signal
#
# {class}`.UncertaintyExtractor` runs the head and converts the class scores, which here are logits, into one normalized
# entropy for each image. Normalized entropy runs from 0, a confident prediction, to 1, a prediction that spreads evenly
# over the six families.

# %%
uncertainty = UncertaintyExtractor(TorchExtractor(head), preds_type="logits", normalize=True)
reference_entropy = np.asarray(uncertainty(embeddings[reference_idx]))
print(f"reference entropy: median {np.median(reference_entropy):.2f}, from {len(reference_entropy)} images")

# %% [markdown]
# ## Monitor with a KS test
#
# {class}`.DriftUnivariate` with `method="ks"` compares the distribution of entropy in a batch of new images with the
# reference. A KS test is calibrated: on a batch from the same distribution it raises an alarm about as often as its
# significance level, 5%. You measure that on batches of 400 operational images and do not assume it.

# %%
BATCH = 400
monitor = DriftUnivariate(method="ks").fit(reference_entropy)
length_monitor = DriftUnivariate(method="ks").fit(np.linalg.norm(embeddings[reference_idx], axis=1)[:, None])

clean_entropy = np.asarray(uncertainty(embeddings[operational_idx]))
batches = [rng.choice(len(operational_idx), BATCH, replace=False) for _ in range(100)]
alarms = [monitor.predict(clean_entropy[batch]).drifted for batch in batches]
print(f"false alarms on {len(batches)} clean batches: {np.mean(alarms):.0%}")


# %% [markdown]
# The significance level promises an alarm on about 5% of clean batches. The rate here can differ, because the 100
# batches overlap and share one pool of 1,500 images, so the rate is only as steady as that pool. Treat the 5% as a
# level that you check, and read every alarm rate below against the rate that you measured.

# %% [markdown]
# ## Degrade the images by known amounts
#
# Now stress the monitor. Add Gaussian noise to the operational images at four strengths, where the strength is the
# standard deviation on a 0 to 1 scale, and blur them at four strengths, where the strength is the width of the blur in
# pixels. You can score these images with labels because you know them. In deployment you would not have the labels, and
# the question is whether the entropy tells you what the accuracy would.


# %%
class Degraded:
    """The operational images, each passed through a degradation."""

    metadata = DatasetMetadata(id="degraded")

    def __init__(self, degrade):
        self.degrade = degrade

    def __len__(self):
        return len(operational_idx)

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        index = int(operational_idx[i])
        return self.degrade(photo(index), index), np.zeros(1, dtype=np.float32), {"id": index}


def noise(strength):
    """Add Gaussian noise, with a different draw for each image."""

    def degrade(image, index):
        draw = np.random.default_rng([int(strength * 1000), index]).normal(0, strength * 255, image.shape)
        return np.clip(image + draw, 0, 255).astype(np.uint8)

    return degrade


def blur(width):
    """Blur each color channel with a Gaussian of the given width in pixels."""
    return lambda image, index: (
        gaussian_filter(image.astype(np.float32), sigma=(0, width, width)).clip(0, 255).astype(np.uint8)
    )


conditions = {f"Noise {s}": noise(s) for s in (0.05, 0.1, 0.2, 0.4)} | {f"Blur {w}": blur(w) for w in (1, 2, 4, 8)}
degraded = {name: np.asarray(Embeddings(Degraded(degrade), extractor=backbone)) for name, degrade in conditions.items()}
print(f"embedded {len(degraded)} degraded copies of the {len(operational_idx):,} operational images")

# %% [markdown]
# Show one operational image at each strength, so that you can see what the monitor is asked to notice.

# %%
picture = Degraded(lambda image, index: image)[5][0]
fig, axes = plt.subplots(1, len(conditions) + 1, figsize=(16, 2.4))
for ax, (name, degrade) in zip(axes, {"Original": lambda image, index: image, **conditions}.items(), strict=True):
    ax.imshow(np.transpose(degrade(picture, 5), (1, 2, 0)))
    ax.set_title(name, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare the alarms with the accuracy
#
# For each condition, draw the same 100 batches of operational images, and record three things for every batch: the
# accuracy of the classifier, which you know here and would not know in deployment, the median entropy, and whether the
# KS monitor raised an alarm. The last column gives the alarm rate of a plain KS test on the length of the embeddings.
# That is a feature check that does not use the classifier at all. A class-mix row builds batches of which 60% come
# from one family, with the same images and no degradation.

# %%
predicted = {"Clean": classifier.predict(embeddings[operational_idx])}
embedded = {"Clean": embeddings[operational_idx]} | degraded
entropy = {"Clean": clean_entropy} | {name: np.asarray(uncertainty(values)) for name, values in degraded.items()}
truth = family[operational_idx]

rows = []
for name, values in embedded.items():
    predictions = classifier.predict(values)
    for batch in batches:
        rows.append({
            "condition": name,
            "accuracy": float(np.mean(predictions[batch] == truth[batch])),
            "median entropy": float(np.median(entropy[name][batch])),
            "entropy alarm": monitor.predict(entropy[name][batch]).drifted,
            "length alarm": length_monitor.predict(np.linalg.norm(values[batch], axis=1)[:, None]).drifted,
        })

big = [f for f in range(len(family_names)) if (truth == f).sum() >= 300]
clean_predictions = predicted["Clean"]
for _ in batches:
    favored = int(rng.choice(big))
    batch = np.concatenate([
        rng.choice(np.flatnonzero(truth == favored), int(0.6 * BATCH), replace=False),
        rng.choice(np.flatnonzero(truth != favored), BATCH - int(0.6 * BATCH), replace=False),
    ])
    rows.append({
        "condition": "Class mix",
        "accuracy": float(np.mean(clean_predictions[batch] == truth[batch])),
        "median entropy": float(np.median(clean_entropy[batch])),
        "entropy alarm": monitor.predict(clean_entropy[batch]).drifted,
        "length alarm": length_monitor.predict(
            np.linalg.norm(embeddings[operational_idx][batch], axis=1)[:, None]
        ).drifted,
    })

batch_results = pl.DataFrame(rows)
summary = batch_results.group_by("condition", maintain_order=True).agg(
    pl.col("accuracy").mean().mul(100).round(0).alias("accuracy (%)"),
    pl.col("median entropy").median().round(2),
    pl.col("entropy alarm").mean().mul(100).round(0).alias("entropy alarms (%)"),
    pl.col("length alarm").mean().mul(100).round(0).alias("length alarms (%)"),
)
print(summary)

# %% [markdown]
# The table holds four findings:
#
# - The Clean row is the false alarm rate. Read every other alarm rate against it.
# - The monitor flags moderate degradation reliably. Noise of 0.1 or more and blur of 2 pixels or more raise an alarm on
#   at least 95% of the batches, and the accuracy there has fallen by at least 9 points. It flags light degradation only
#   some of the time: noise of 0.05 on about three batches in four, and blur of 1 pixel on about one in six.
# - The entropy rises with the degradation, but its size does not tell you the loss in accuracy. From blur of 4 pixels
#   to blur of 8 pixels the median entropy stays near 0.56 while the accuracy keeps falling. A badly degraded image can
#   make the classifier confidently wrong.
# - The entropy monitor is neither the most sensitive check nor specific to degradation. A KS test on the length of the
#   embeddings, which does not use the classifier, raises an alarm on every batch of every degraded condition, including
#   blur of 1 pixel, where the entropy monitor flags about one batch in six. The entropy monitor also raises alarms on
#   most of the class-mix batches, where the accuracy did not fall, and the length test does not. The two checks answer
#   different questions. The length test asks whether the images changed, and the entropy monitor asks whether the
#   confidence of the classifier changed. Neither tells you that the accuracy fell.

# %%
(
    px.scatter(
        batch_results.with_columns(
            pl
            .col("condition")
            .str.split(" ")
            .list.first()
            .replace({"Clean": "Clean", "Class": "Class mix"})
            .alias("shift")
        ),
        x="median entropy",
        y="accuracy",
        color="shift",
        hover_data=["condition"],
        opacity=0.6,
        height=450,
    ).update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Accuracy of a batch (needs labels)",
        xaxis_title="Median entropy of a batch",
    )
)

# %% [markdown]
# Each point is a batch. Across the clean, class-mix, noise and blur batches the points follow one downward trend: a
# batch with a lower accuracy has a higher median entropy. That is the case for a monitor of this kind. The trend breaks
# at the heavy end. Batches near 35% accuracy have a median entropy near 0.71 under noise and near 0.56 under blur, so
# for badly degraded images you cannot read the accuracy of a batch from its entropy.
#
# The class-mix batches lie on the trend in the other direction. They sit mostly at a lower entropy and a higher
# accuracy than the clean batches, because a batch with many images of one family changes the mix of easy and hard
# images. The monitor raises an alarm there on a change that is not a loss.

# %% [markdown]
# ## A family the classifier never saw
#
# The images of the unseen family come from vehicles that none of the six families describes. A good monitor would
# raise an alarm when they arrive. Build batches in which a growing share of the images come from the unseen family, and
# record how often the monitor raises an alarm. {class}`.ClasswiseUncertaintyExtractor` groups the unseen images by the
# family that the classifier predicts, which shows where they go.

# %%
unseen_embeddings = embeddings[unseen]
unseen_entropy = np.asarray(uncertainty(unseen_embeddings))
by_family = ClasswiseUncertaintyExtractor(TorchExtractor(head), preds_type="logits", normalize=True, threshold=1.0)(
    unseen_embeddings
)
placed = {family_names[family_id]: len(values) / len(unseen_embeddings) for family_id, values in by_family.items()}
print(
    f"median entropy: unseen family {np.median(unseen_entropy):.2f}, operational images {np.median(clean_entropy):.2f}"
)
print("predicted family of the unseen images:")
for name, share in sorted(placed.items(), key=lambda item: -item[1]):
    print(f"  {name:<26}{share:>5.0%}")

mix_alarms = {}
for share in (0.0, 0.1, 0.25, 0.5, 1.0):
    n_new = int(share * BATCH)
    outcomes = []
    for batch in batches:
        new = unseen_entropy[rng.choice(len(unseen_entropy), n_new, replace=False)]
        outcomes.append(monitor.predict(np.concatenate([clean_entropy[batch[: BATCH - n_new]], new])).drifted)
    mix_alarms[share] = float(np.mean(outcomes))
print("\nshare of a batch from the unseen family -> alarms")
for share, rate in mix_alarms.items():
    print(f"  {share:>4.0%} -> {rate:>4.0%}")

# %% [markdown]
# The classifier files most of the unseen images under self-propelled artillery, and it is more confident on them than
# on the operational images, with a median entropy of 0.36 against 0.44. So the entropy moves down and not up. The
# monitor reacts to a change in the distribution of entropy in either direction, and here the change stays small until
# the unseen family is a large share of the batch. A share of 10% or 25% raises alarms at about the clean rate, 50%
# raises them on about a third of the batches, and only a batch made entirely of the unseen family raises them almost
# every time. Do not rely on a monitor of this kind to catch a new type of vehicle.

# %% [markdown]
# ## An overconfident classifier
#
# You chose strong regularization for the head. Fit a second head with the default, weaker regularization (`C=1.0`) on
# the same images, build the same monitor from its entropy, and run it on the same batches. This head fits its training
# images more tightly, and it is more sure of its answers.

# %%
loose = LogisticRegression(C=1.0, max_iter=3000).fit(embeddings[train_idx], family[train_idx])
loose_uncertainty = UncertaintyExtractor(TorchExtractor(to_torch(loose)), preds_type="logits", normalize=True)
loose_reference = np.asarray(loose_uncertainty(embeddings[reference_idx]))
loose_monitor = DriftUnivariate(method="ks").fit(loose_reference)

show = ["Clean", "Noise 0.05", "Noise 0.1", "Blur 2"]
alarm_rates = {
    "C=0.05 (used above)": [
        float(batch_results.filter(pl.col("condition") == name)["entropy alarm"].to_numpy().mean()) for name in show
    ],
    "C=1.0": [
        float(
            np.mean([
                loose_monitor.predict(np.asarray(loose_uncertainty(embedded[name]))[batch]).drifted for batch in batches
            ])
        )
        for name in show
    ],
}
heads = {}
print(f"{'head':<22}{'accuracy':>9}{'at Blur 2':>11}{'median entropy':>16}" + "".join(f"{name:>12}" for name in show))
for (label, rates), model in zip(alarm_rates.items(), (classifier, loose), strict=True):
    heads[label] = {
        "accuracy": model.score(embeddings[operational_idx], truth),
        "blur accuracy": model.score(embedded["Blur 2"], truth),
        "median entropy": float(np.median(reference_entropy if model is classifier else loose_reference)),
        "alarms": dict(zip(show, rates, strict=True)),
    }
    row = heads[label]
    print(
        f"{label:<22}{row['accuracy']:>9.0%}{row['blur accuracy']:>11.0%}{row['median entropy']:>16.2f}"
        + "".join(f"{rate:>12.0%}" for rate in rates)
    )

# %% [markdown]
# The second head is less accurate and far more confident, with a median reference entropy of 0.15 against 0.43. Its
# monitor raises fewer false alarms and it misses a blur of 2 pixels almost entirely, on 4% of the batches against 95%
# for the first head, although the accuracy of this head fell on those images too. Entropy has little room to rise when
# a classifier is already sure of almost everything, so a monitor built on an overconfident model can be blind to a
# shift that would otherwise raise an alarm.
#
# That is why you check a monitor before you rely on it. Degrade labeled images by known amounts, as you did here, and
# confirm that the monitor of your classifier raises alarms on the shifts that you care about.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# The data and the three sets: families of repeats stay on one side
assert len(known) == 8184
assert unseen.sum() == 1260
assert len(family_names) == 6
sets = [set(repeat_family[idx]) for idx in (train_idx, reference_idx, operational_idx)]
assert not (sets[0] & sets[1])
assert not (sets[0] & sets[2])
assert not (sets[1] & sets[2])
assert 2500 < len(train_idx) < 3500
assert 1500 < len(reference_idx) < 2500
assert len(operational_idx) == 1500
assert 0.55 < classifier.score(embeddings[operational_idx], truth) < 0.72

# The false alarm rate on clean batches is near the 5% level
assert 0.0 <= np.mean(alarms) < 0.2

# The table the text describes
by = {row["condition"]: row for row in summary.iter_rows(named=True)}
accuracy = {name: row["accuracy (%)"] for name, row in by.items()}
median_entropy = {name: row["median entropy"] for name, row in by.items()}
entropy_alarm = {name: row["entropy alarms (%)"] for name, row in by.items()}
length_alarm = {name: row["length alarms (%)"] for name, row in by.items()}
assert (
    accuracy["Clean"] >= accuracy["Noise 0.05"] > accuracy["Noise 0.1"] > accuracy["Noise 0.2"] >= accuracy["Noise 0.4"]
)
assert accuracy["Clean"] >= accuracy["Blur 1"] > accuracy["Blur 2"] > accuracy["Blur 4"] > accuracy["Blur 8"]
moderate = ["Noise 0.1", "Noise 0.2", "Noise 0.4", "Blur 2", "Blur 4", "Blur 8"]
assert all(entropy_alarm[name] >= 90 for name in moderate)
assert all(accuracy["Clean"] - accuracy[name] >= 7 for name in moderate)
assert 30 <= entropy_alarm["Noise 0.05"] < 100
assert entropy_alarm["Blur 1"] < 50
assert (
    median_entropy["Noise 0.05"]
    < median_entropy["Noise 0.1"]
    < median_entropy["Noise 0.2"]
    < median_entropy["Noise 0.4"]
)
assert median_entropy["Blur 8"] < median_entropy["Blur 4"] + 0.03
assert accuracy["Blur 8"] < accuracy["Blur 4"] - 4
assert all(length_alarm[name] >= 90 for name in conditions)
assert entropy_alarm["Blur 1"] < length_alarm["Blur 1"]
assert entropy_alarm["Class mix"] > 40
assert length_alarm["Class mix"] < 30
assert accuracy["Class mix"] >= accuracy["Clean"] - 5

# Accuracy and entropy follow one downward trend across the batches, and the class-mix batches sit at a lower entropy
assert batch_results.select(pl.corr("median entropy", "accuracy", method="spearman")).item() < -0.6
assert median_entropy["Class mix"] < median_entropy["Clean"]

# The noise and blur batches do not share one curve
assert median_entropy["Noise 0.4"] - median_entropy["Blur 8"] > 0.08
assert abs(accuracy["Noise 0.4"] - accuracy["Blur 8"]) < 6

# The unseen family
assert placed["Self Propelled Artillery"] > 0.6
assert np.median(unseen_entropy) < np.median(clean_entropy)
assert mix_alarms[0.1] < 0.3
assert mix_alarms[0.25] < 0.35
assert 0.1 < mix_alarms[0.5] < 0.6
assert mix_alarms[1.0] > 0.85

# The overconfident head
strong, weak = heads["C=0.05 (used above)"], heads["C=1.0"]
assert weak["accuracy"] < strong["accuracy"]
assert weak["median entropy"] < 0.6 * strong["median entropy"]
assert strong["alarms"]["Blur 2"] > 0.8
assert weak["alarms"]["Blur 2"] < 0.3
assert weak["blur accuracy"] < weak["accuracy"] - 0.05

# %% [markdown]
# ## Conclusion
#
# You trained a vehicle-family classifier and built a label-free monitor from its own uncertainty with
# {class}`.UncertaintyExtractor` and {class}`.DriftUnivariate`. You measured its false alarm rate on clean batches,
# degraded images by known amounts to compare its alarms with the accuracy that you would not see in deployment, and
# tested three things that it misses.
#
# You learned five things:
#
# - Prediction entropy is a label-free drift signal, and a KS test on it is a calibrated monitor whose false alarm rate
#   you can measure.
# - The monitor flags moderate degradation reliably and light degradation only some of the time.
# - The size of the entropy shift does not tell you the loss in accuracy, and a badly degraded image can make a
#   classifier confidently wrong.
# - Use the monitor next to a feature check and not instead of one. A KS test on the length of the embeddings flagged
#   the degradation more often, and the entropy monitor reacted to a change in the mix of families that did not cost
#   accuracy.
# - A new family of vehicles, and an overconfident classifier, can leave the monitor quiet. Check the monitor with
#   degraded labeled images before you rely on it.

# %% [markdown]
# ## Next steps
#
# - [Taxonomy of shift](../concepts/DistributionShift.md#taxonomy-of-shift) — Read about shift types and when
#   uncertainty-based detection is and is not the right tool.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Compare prediction uncertainty with feature-based
#   drift detection on dataset embeddings.
# - [Detect uncertainty drift with a MAITE model](./h2_detect_uncertainty_drift.py) — Test the entropy of a MAITE
#   model's predictions for drift.

# %% [markdown]
# ## On your own
#
# Once you are comfortable with the workflow, adapt it to your own deployed model:
#
# - Swap the classifier for your own, and take the logits that it produces for each image
# - Degrade a labeled sample of your own images by the amounts that you expect in the field, and check the alarms before
#   you trust the monitor
# - Compare the alarms with a feature check on your own embeddings, as in the table above
# - Try other significance levels and batch sizes to trade false alarms against missed shifts
