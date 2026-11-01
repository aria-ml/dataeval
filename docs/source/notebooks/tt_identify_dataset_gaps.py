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
# # Identify gaps in a labeled dataset
#
# In this tutorial you will combine an ontology, class labels, and image embeddings to find gaps in a dataset that no
# single measurement shows on its own. You will work with MilitaryVehicles, a collection of photographs of 24 types of
# military land vehicle, and with real embeddings from a pretrained ResNet18.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [Data Scientist](../getting-started/roles/data_scientist.md), [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you will do
#
# - Read flat per-class counts with {func}`.label_stats`
# - Build an {class}`.Ontology` of the classes a model is meant to recognize from the hierarchy that the dataset ships
#   with
# - Measure label-space coverage with {class}`.Representation`. This shows which classes have no examples and which are
#   under-represented.
# - Simulate three quick fixes that bring thin classes up to a typical class size
# - Extract image embeddings with a pretrained ResNet18
# - Measure embedding-space coverage with {class}`.Coverage`. This shows which classes the quick fixes padded without
#   adding variety: clustered, one-dimensional, or padded with duplicate frames.
# - Read the three together to choose a fix for each gap
#
# ## What you will learn
#
# - You will learn that "is my dataset complete?" has two orthogonal axes: which categories you have (labels) and how
#   varied each one is (embeddings). You need both.
# - You will learn that reaching a target count does not close a gap. A class can reach its count and still lack
#   variety.

# %% [markdown]
# ## What you will need
#
# - Basic familiarity with Python
# - A Python environment with `dataeval`, `dataeval-plots`, `maite-datasets`, and `torchvision` installed

# %% [markdown]
# ## Introduction
#
# A dataset is complete relative to what your model must recognize. It can fall short in two independent ways:
#
# - **Label-space gaps** occur when a sanctioned class has no examples, or a whole branch of the taxonomy is empty. Flat
#   counts cannot show this: a class with zero examples contributes zero rows, so it does not appear at all. An
#   **ontology** supplies the list of what should exist, and {class}`.Representation` measures coverage against it.
# - **Embedding-space gaps** occur when a class is present and well counted, but its images lack variety.
#   {class}`.Coverage` measures this from image embeddings along three independent axes: **dispersion** (is the class
#   clustered into a tiny region?), **isotropy** (does it vary in many directions, or just one?), and **near-duplicate
#   fraction** (is a share of the class the same frame repeated?).
#
# Counts are the usual measure of progress in a collection effort, which makes them easy to satisfy without adding
# anything new. You will audit a real collection, act on the audit the quick way, and then check whether the fixes
# added any variety.

# %% [markdown]
# ## Setup
#
# Begin by importing the pieces you need. You will load the MilitaryVehicles images and extract embeddings from them
# with a pretrained ResNet18. See [Assess an unlabeled data space](./tt_assess_data_space.py) for a deeper look at the
# embedding-only gap workflow with clustering and outliers.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass

# %%
import io
from collections.abc import Sequence

import dataeval_plots as dep
import numpy as np
import plotly.io as pio
import polars as pl
import torch
from maite_datasets.image_classification import MilitaryVehicles
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata, Ontology
from dataeval.config import set_device
from dataeval.core import label_stats
from dataeval.extractors import TorchExtractor
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.scope import Coverage, Representation

pl.Config.set_tbl_rows(30)  # show every class in the tables below, not a truncated view
pl.Config.set_tbl_width_chars(120)  # keep the flag columns on one line

# Use the GPU if one is available, otherwise the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

# Render the embedding plots below as interactive plotly figures.
dep.set_default_backend("plotly")
pio.renderers.default = "notebook"  # embed the plotly JS in the notebook output

# %% [markdown]
# ## The dataset
#
# Load the train split of MilitaryVehicles with `MilitaryVehicles` from `maite_datasets`. It has 7,823 photographs of 24
# types of military land vehicle, from tanks to self-propelled artillery. Treat it as a collection that was gathered for
# a vehicle recognizer. Start with {func}`.label_stats`, which gives the per-class counts.

# %%
dataset = MilitaryVehicles(root="./data", image_set="train", download=True)
metadata = Metadata(dataset)

stats = label_stats(metadata.class_labels, index2label=metadata.index2label)
counts = {metadata.index2label[i]: c for i, c in stats["label_counts_per_class"].items()}
counts_table = pl.DataFrame({"class": list(counts), "count": list(counts.values())}).sort("count")

print("images:", len(dataset), "| classes present:", stats["class_count"])
print(counts_table.head(6))

# %% [markdown]
# The counts look healthy. Every class except `30N6E` holds at least 240 images, and `30N6E` is the only thin class.
# Counts cannot show two things. A class with no images contributes no rows, so it does not appear in the table. And
# nothing here tells you whether the images of a class are varied. The next sections address both.

# %% [markdown]
# ## The ontology
#
# Now state what the recognizer is meant to know. The dataset ships with a taxonomy as its `hierarchy` attribute. It
# resolves land vehicles to seven families, such as tanks and self-propelled artillery, and it sanctions two more
# branches that have no photographs in the dataset: `watercraft` and `aircraft`. Build the {class}`.Ontology` from it
# with {meth}`.Ontology.from_hierarchy`.

# %%
ontology = Ontology.from_hierarchy(dataset.hierarchy)
print(ontology)
print("sanctioned leaf classes:", len(ontology.leaves))

# %% [markdown]
# ## Label-space coverage
#
# Now run {class}`.Representation`, which compares the labels to the ontology and returns a collection worklist: which
# classes to acquire (none collected) or augment (under-represented), relative to an even spread across the sanctioned
# classes.

# %%
representation = Representation(ontology).evaluate(metadata)

print(f"leaf coverage: {representation.leaf_coverage:.0%} of sanctioned classes have any examples")
print(representation.data().select(["concept", "parent", "action", "count", "target", "deficit"]))

# %% [markdown]
# The ontology shows what the counts did not. `aircraft` and `watercraft` are sanctioned but have no images, so the
# worklist marks them to acquire. Nine more classes fall below the even-spread target, so the worklist marks them to
# augment. `30N6E` is furthest from its target at 182 images short, followed by `BTR-70` at 61 and `Iskander` at 49.

# %% [markdown]
# ## Act on the worklist
#
# Acquiring images of aircraft and watercraft is a new collection effort, so suppose the team starts with the thin
# classes it can fix quickly. It closes the three largest shortfalls of the worklist with three common shortcuts, and
# each shortcut adds images without adding variety:
#
# - `30N6E`: the team films one vehicle and pulls 182 frames from the clip. The frames differ only by a little camera
#   shake and drift.
# - `BTR-70`: the team films one vehicle while the camera zooms in, and pulls 120 frames from the sweep.
# - `Iskander`: the team ingests 50 existing images a second time. The system that delivered them saved each one again
#   as a JPEG file, so every copy is near-identical to its original.
#
# You simulate each shortcut with images that are already in the dataset, so that the tutorial stays self-contained.
# The `crop` helper cuts a window out of an image and resizes it back to the original size, which is what camera shake
# and a zoom do to a frame. The `resave` helper saves an image as a JPEG file and loads it again.

# %%
rng = np.random.default_rng(0)
by_class = {name: np.where(metadata.class_labels == index)[0] for index, name in metadata.index2label.items()}


def crop(image: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """Cut a window out of a channels-first image and resize it back to the original size."""
    full_height, full_width = image.shape[1:]
    window = np.moveaxis(image, 0, -1)[top : top + height, left : left + width]
    resized = Image.fromarray(window).resize((full_width, full_height), Image.Resampling.BILINEAR)
    return np.moveaxis(np.asarray(resized), -1, 0)


def resave(image: np.ndarray) -> np.ndarray:
    """Save an image as a JPEG file and load it again, as a copy that passed through another system would be."""
    buffer = io.BytesIO()
    Image.fromarray(np.moveaxis(image, 0, -1)).save(buffer, format="JPEG", quality=90)
    return np.moveaxis(np.asarray(Image.open(buffer)), -1, 0)


def one_image(name: str) -> np.ndarray:
    """A random image of a class."""
    return np.asarray(dataset[int(rng.choice(by_class[name]))][0])


# `30N6E`: 182 frames from one clip, which differ only by a random shift and zoom of a few percent.
image = one_image("30N6E")
height, width = image.shape[1:]
clip = []
for _ in range(182):
    window_h, window_w = (int(size * rng.uniform(0.94, 1.0)) for size in (height, width))
    top, left = int(rng.integers(0, height - window_h + 1)), int(rng.integers(0, width - window_w + 1))
    clip.append(crop(image, top, left, window_h, window_w))

# `BTR-70`: 120 frames from one clip in which the camera zooms from the full frame to half of it.
image = one_image("BTR-70")
height, width = image.shape[1:]
zoom = []
for scale in np.linspace(1.0, 0.5, 120):
    window_h, window_w = int(height * scale), int(width * scale)
    zoom.append(crop(image, (height - window_h) // 2, (width - window_w) // 2, window_h, window_w))

# `Iskander`: 50 existing images ingested a second time, each saved again as a JPEG file.
copies = [resave(np.asarray(dataset[int(i)][0])) for i in rng.choice(by_class["Iskander"], size=50, replace=False)]

quick_fixes = {"30N6E": clip, "BTR-70": zoom, "Iskander": copies}

# %% [markdown]
# Wrap the original images and the new frames into one dataset. The new frames follow the original images and carry
# the label of the class they were added to.


# %%
class ToppedUpDataset:
    """The original images followed by extra images, each added to one class."""

    def __init__(self, base: MilitaryVehicles, extras: Sequence[tuple[np.ndarray, int]]) -> None:
        self._base = base
        self._extras = list(extras)
        self.metadata = DatasetMetadata(id="MilitaryVehicles_topped_up", index2label=base.index2label)

    def __len__(self) -> int:
        return len(self._base) + len(self._extras)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        if index < len(self._base):
            return self._base[index]
        image, label = self._extras[index - len(self._base)]
        target = np.zeros(len(self._base.index2label), dtype=np.float32)
        target[label] = 1.0
        return image, target, {"id": f"added_{index}"}


extras = [(frame, dataset.label2index[name]) for name, frames in quick_fixes.items() for frame in frames]
topped_up = ToppedUpDataset(dataset, extras)
topped_up_metadata = Metadata(topped_up)

topped_up_stats = label_stats(topped_up_metadata.class_labels, index2label=topped_up_metadata.index2label)
topped_up_counts = {topped_up_metadata.index2label[i]: c for i, c in topped_up_stats["label_counts_per_class"].items()}
print(
    pl.DataFrame({
        "class": list(quick_fixes),
        "before": [counts[name] for name in quick_fixes],
        "after": [topped_up_counts[name] for name in quick_fixes],
    })
)

# %% [markdown]
# All three classes now hold about 300 images or more, in line with the other classes. An audit that relies on counts
# would call these three gaps closed.

# %% [markdown]
# ## Embedding extraction
#
# To measure variety you need a numeric representation of each image. Extract a feature vector per image with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
# Torchvision. Point a {class}`.TorchExtractor` at its penultimate `avgpool` layer (its learned features, not the
# 1000-class logits), give it the model's standard resize-and-normalize preprocessing as `transforms`, and run the
# dataset through {class}`.Embeddings`. That yields a 512-dimensional feature per image.
#
# For other ways to build embeddings, see [Create embeddings](./h2_create_embeddings.py).

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
extractor = TorchExtractor(resnet, transforms=ResNet18_Weights.DEFAULT.transforms(), layer_name="avgpool")
features = np.asarray(Embeddings(topped_up, extractor=extractor, batch_size=64))
print("feature shape:", features.shape)

# %% [markdown]
# ### Reduce to 128 dimensions with PCA
#
# {class}`.Coverage` estimates isotropy (in how many independent directions a class varies) from an effective-dimension
# calculation that is only well defined when a class has more samples than embedding dimensions. Every class here has
# a few hundred images, so at 512 dimensions every class falls short and isotropy comes back null. Reducing to 128
# dimensions with PCA, which keeps the highest-variance axes, brings the dimension count below the size of every class,
# so isotropy becomes measurable. Fit PCA once across the whole dataset, never per class.

# %%
embeddings = PCA(n_components=128, random_state=0).fit_transform(features)
print("embedding shape:", embeddings.shape)

# %% [markdown]
# :::{note}
#
# ResNet18 keeps this tutorial fast. If you need sharper separation between varied and near-duplicate classes on your
# own data, swap in a larger backbone such as `torchvision.models.resnet50`. The rest of the workflow is unchanged.
#
# :::

# %% [markdown]
# ## Embedding-space coverage
#
# Now run {class}`.Coverage`, which describes each class with three independent variety signals. Each is a gap that
# raw counts cannot see:
#
# - `dispersion`: the magnitude of spread (mean distance to centroid, relative to a typical class). A value near 1 is
#   typical, and a value well below 1 means the class is clustered.
# - `isotropy`: the shape of that spread (in how many independent directions the class varies). A value well below the
#   typical class means it varies along one axis, even when dispersion is fine.
# - `near_duplicate_fraction`: redundancy (the share of the class's images whose nearest neighbor is unusually close,
#   compared with a typical class). A high value means a share of the class is repeated frames, even when dispersion
#   is fine.
#
# {class}`.Coverage` also turns each signal into a flag per class: `is_clustered`, `is_one_dimensional`, and
# `has_near_duplicates`. By default a flag is raised only when a class has collapsed to half of a typical class, which
# keeps the flags quiet on healthy data. The classes of this dataset are tight, so set `dispersion_factor` and
# `isotropy_factor` to 0.8 to flag a class that is a fifth below its peers. Print the flagged classes, and summarize
# the classes that no flag marks.

# %%
coverage = Coverage(num_observations=20, min_class_samples=20, dispersion_factor=0.8, isotropy_factor=0.8).evaluate(
    topped_up_metadata, embeddings=embeddings
)

signals = ["dispersion", "isotropy", "near_duplicate_fraction"]
flags = ["is_clustered", "is_one_dimensional", "has_near_duplicates"]
cov = coverage.data()
print(cov.filter(pl.any_horizontal(flags)).select(["class", *signals, *flags]))
print(cov.filter(~pl.any_horizontal(flags)).select(signals).describe())

# %% [markdown]
# The flags mark the three classes that received a quick fix. The other 21 classes behave alike. Their
# dispersion stays close to 1, their isotropy stays in a narrow band, and almost none of their images have a
# near-identical neighbor. Each flagged class raises a different set of flags:
#
# - `30N6E` raises all three. Its dispersion is about 0.7 where the other classes have 0.9 or more, and its isotropy is
#   about 0.2 where they have 0.35 or more. Its 182 clip frames collapse into a tiny region of the space, so most of
#   the class sits in one spot. About 0.6 of its images have a near-identical neighbor.
# - `BTR-70` raises `is_one_dimensional` and `has_near_duplicates`. Its dispersion is normal, its isotropy is about 0.3,
#   and about a third of its images have a near-identical neighbor. The zoom sweep spreads out along a single
#   direction, and neighboring frames of the sweep sit close together.
# - `Iskander` raises only `has_near_duplicates`. Its dispersion and isotropy are normal, and about a third of its
#   images have a near-identical neighbor, where the other classes have almost none. The 50 copies and the 50 originals
#   they copy make up a third of the class.
#
# `has_near_duplicates` is raised for all three, because each shortcut repeats frames, but it does not say how the class
# repeats itself. `is_clustered` tells you that a class collapsed into one region, and `is_one_dimensional` tells you
# that it stretched along one axis. Read the flags together. A class that raises only `has_near_duplicates` holds
# repeated frames. A class that raises the other flags as well lacks variety in a way that duplicates alone do not
# explain.

# %% [markdown]
# ### See the shapes
#
# You can see two of the three gaps. Project the embeddings of the three classes and of three healthy classes down to
# 2D with PCA, and color by class. The healthy classes spread into broad overlapping clouds. The `30N6E` frames form a
# single tight knot away from the rest, and the `BTR-70` zoom frames gather into a denser band within their class. The
# `Iskander` copies do not show. A copy lies on top of its original, so the plot cannot tell a copy from the image it
# duplicates, and only the near-duplicate fraction reveals them. Hover and zoom to explore.
#
# PCA is the right choice here because it preserves relative spread, so a tight class still looks tight.
# Neighbor-embedding methods like t-SNE or UMAP renormalize local density and would mask the effects you want to see.

# %%
shown = [*quick_fixes, "T-72", "BMP-2", "BTR-80"]
in_view = np.isin(topped_up_metadata.class_labels, [dataset.label2index[name] for name in shown])
dep.project(
    embeddings[in_view],
    method="pca",
    labels=topped_up_metadata.class_labels[in_view],
    label_names=topped_up_metadata.index2label,
    title="MilitaryVehicles embeddings in 2D (PCA) after the quick fixes",
)

# %% [markdown]
# ## Choose a fix for each gap
#
# The worklist and the coverage flags hold all the evidence, and each one points to a different fix:
#
# | Evidence | What it means | Fix |
# | --- | --- | --- |
# | `acquire` in the worklist | The class is sanctioned and has no images | Acquire images of it |
# | `augment` in the worklist | The class has fewer images than an even spread | Collect more images of it |
# | `is_clustered` | The class collapsed into one small region | Broaden it with images that differ |
# | `is_one_dimensional` | The class varies along one axis only | Vary the images along other axes |
# | `has_near_duplicates` | A share of the class repeats frames | Remove the repeats and collect new images |
#
# Read the collection plan for this dataset off the worklist and the flags:
#
# - Acquire `aircraft` and `watercraft`, which are sanctioned and have no images.
# - Collect more of the thin classes that the quick fixes did not touch. `BM-30`, `Pantsir-S1`, and `T-62` are the
#   thinnest.
# - Broaden `30N6E`, which `is_clustered` shows collapsed into one region.
# - Vary `BTR-70` along other axes, because `is_one_dimensional` shows that the zoom sweep varies along one.
# - Deduplicate `Iskander`, which only `has_near_duplicates` marks.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Lock the claims this tutorial makes in prose, so a dependency or data change cannot
# silently invert them while the narrative keeps asserting the old result.

# The dataset and its counts
assert (len(dataset), len(dataset.index2label)) == (7823, 24)
assert counts["30N6E"] == 119
assert min(count for name, count in counts.items() if name != "30N6E") >= 240

# The ontology and the first worklist
assert len(ontology.leaves) == 26
assert round(representation.leaf_coverage, 2) == 0.92
first = {row["concept"]: row for row in representation.data().iter_rows(named=True)}
assert {name for name, row in first.items() if row["action"] == "acquire"} == {"aircraft", "watercraft"}
assert sum(row["action"] == "augment" for row in first.values()) == 9
assert (first["30N6E"]["deficit"], first["BTR-70"]["deficit"], first["Iskander"]["deficit"]) == (182, 61, 49)
assert sorted(first, key=lambda name: first[name]["deficit"], reverse=True)[2:5] == ["30N6E", "BTR-70", "Iskander"]

# The quick fixes
assert [len(frames) for frames in quick_fixes.values()] == [182, 120, 50]
assert [topped_up_counts[name] for name in quick_fixes] == [301, 360, 302]
assert all(topped_up_counts[name] >= 300 for name in quick_fixes)

# The sizes the text quotes
coverage_by_class = {row["class"]: row for row in cov.iter_rows(named=True)}
peers = [row for name, row in coverage_by_class.items() if name not in quick_fixes]
assert len(peers) == 21
assert all(0.9 <= row["dispersion"] <= 1.1 for row in peers)
assert all(0.35 <= row["isotropy"] <= 0.5 for row in peers)
assert all(row["near_duplicate_fraction"] < 0.05 for row in peers)
assert 0.6 < coverage_by_class["30N6E"]["dispersion"] < 0.8
assert 0.1 < coverage_by_class["30N6E"]["isotropy"] < 0.3
assert 0.4 < coverage_by_class["30N6E"]["near_duplicate_fraction"] < 0.8
assert 0.2 < coverage_by_class["BTR-70"]["isotropy"] < 0.34
assert 0.9 <= coverage_by_class["BTR-70"]["dispersion"] <= 1.1
assert 0.2 < coverage_by_class["Iskander"]["near_duplicate_fraction"] < 0.4
assert 0.9 <= coverage_by_class["Iskander"]["dispersion"] <= 1.1
assert 0.35 <= coverage_by_class["Iskander"]["isotropy"] <= 0.5
assert 0.2 < coverage_by_class["BTR-70"]["near_duplicate_fraction"] < 0.4

# The flags
flagged = {row["class"]: row for row in cov.filter(pl.any_horizontal(flags)).iter_rows(named=True)}
assert set(flagged) == set(quick_fixes)
assert tuple(flagged["30N6E"][flag] for flag in flags) == (True, True, True)
assert tuple(flagged["BTR-70"][flag] for flag in flags) == (False, True, True)
assert tuple(flagged["Iskander"][flag] for flag in flags) == (False, False, True)

# The shape the text describes: the clip frames form a knot much tighter than a healthy class
xy = PCA(n_components=2, random_state=0).fit_transform(embeddings[in_view])
shown_labels = topped_up_metadata.class_labels[in_view]
is_new = np.arange(len(topped_up))[in_view] >= len(dataset)  # the new frames follow the original images
clip_spread = xy[is_new & (shown_labels == dataset.label2index["30N6E"])].std(axis=0).mean()
healthy_spread = xy[shown_labels == dataset.label2index["T-72"]].std(axis=0).mean()
assert clip_spread < 0.25 * healthy_spread

# The collection plan
thin = [name for name in sorted(counts, key=lambda name: counts[name]) if name not in quick_fixes]
assert thin[:3] == ["BM-30", "Pantsir-S1", "T-62"]
assert all(first[name]["action"] == "augment" for name in thin[:3])

# %% [markdown]
# Each check caught a gap the others missed:
#
# - Counts alone pointed only to `30N6E`. After the quick fixes they pointed to nothing, even though three classes had
#   gained nothing new.
# - The ontology added `aircraft` and `watercraft`, which have no images and so no rows to count.
# - The embeddings added the three quick fixes: `30N6E` (clustered), `BTR-70` (one-dimensional), and `Iskander`
#   (duplicate-padded). Each one repaired the count and left the variety where it was.
#
# Each gap maps to a different fix: acquire the missing classes, collect more of the thin ones, and replace the padding
# in the three padded classes with new images that differ in the ways the old ones do not.

# %% [markdown]
# ## Conclusion
#
# In this tutorial you turned a set of counts that looked healthy into a concrete collection plan. You saw that "is my
# dataset complete?" is two questions: {class}`.Representation` answered which categories you have against an
# {class}`.Ontology`, and {class}`.Coverage` answered how varied each one is along three independent axes (dispersion,
# isotropy, near-duplicate fraction). You also saw that a count is easy to satisfy without adding variety, which is why
# the second question matters after every collection effort.

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Learn how hierarchies and taxons organize dataset class labels and target
#   scopes.
# - [Dataset bias and coverage](../concepts/DatasetBias.md) — Read about dispersion, isotropy, and representation
#   metrics for assessing dataset coverage.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Discover coverage gaps, clusters, and outliers in
#   unlabeled dataset embeddings.
# - [Reconcile labels against an ontology](./h2_reconcile_labels_ontology.py) — Map and validate dataset labels
#   against a target domain ontology.

# %% [markdown]
# ## On your own
#
# Swap MilitaryVehicles for your own dataset: build (or load) an {class}`.Ontology` of your sanctioned classes, extract
# embeddings with an extractor, and run the same three checks. The tables you read are your collection plan.
