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
# # Assess whether a classification requirement is reachable
#
# This tutorial uses {class}`~dataeval.performance.Feasibility` to decide whether three performance requirements for a
# military vehicle classifier are theoretically reachable using the `MilitaryVehicles` images.
#
# Estimated time to complete: 30 minutes
#
# Relevant ML stages: [Model Development](../getting-started/roles/ML_Lifecycle.md#model-development)
#
# Relevant personas: [Data Scientist](../getting-started/roles/data_scientist.md),
# [ML Engineer](../getting-started/roles/ml_engineer.md)

# %% [markdown]
# :::{note}
# This tutorial can be followed on its own, but reading the [Performance limits](../concepts/PerformanceLimits.md)
# concept page first is recommended.
# :::

# %% [markdown]
# ## What you will do
#
# - You will ask {class}`~dataeval.performance.Feasibility` whether a model can name the type of military vehicle in a
#   photo well enough (meeting a performance goal), using the `MilitaryVehicles` images, which show 24 vehicle names in
#   7 categories.
# - When the evidence cannot provide a definitive answer, you will change the target task twice: first to naming the
#   vehicle's broader category,
#   then to differentiating only between types of tanks.
# - You will read each answer as a printout and as a chart.

# %% [markdown]
# ## What you will learn
#
# - You will learn how to show a performance requirement met using the lower bound "at least this" number from an
#   existing model.
# - You will learn how to show that a performance requirement is unreachable using the upper bound "no more than this"
#   number.
# - You will learn what to do when the target performance requirement lies between the two known bounds.
# - You will learn how to ask a coarser question of the same model by automatically relabeling the images.
# - You will learn how to adjust the decision rule for the tank class to reduce the number of false positives without
#   retraining the model.

# %% [markdown]
# ## What you will need
#
# - Environment Requirements
#   - `dataeval`
#   - `dataeval-plots`
#   - `maite-datasets`
#   - `scikit-learn`
#   - `torchvision`
# - About 450 MB of downloads: the images (about 100 MB) and a pretrained image model (about 330 MB)
# - A GPU is optional. It makes the embeddings much faster to create

# %% [markdown]
# ## Introduction
#
# Your program is building a system that names the military vehicle in a photo (e.g. T-72 tank or BTR-80 armored
# personnel carrier). The `Feasibility` of this problem might be limited by inherent ambiguity between classes in the
# data, or by the complexity of the model. Before investing heavily into model development, you want to check whether or
# not the data is a limiting factor. You first choose a set of performance targets and ask
# {class}`~dataeval.performance.Feasibility` if each requirement is reasonable given the data. `Feasibility` answers by
# describing each requirement as one of the following:
#
# - **Demonstrated** (`demonstrated_feasible`): A pre-existing model (minimally tuned to your classification problem)
#   can easily achieve your performance requirement. This model is called a "witness", and establishes a
#   **lower confidence bound**
# - **Ruled out** (`infeasible_under_recorded_assumptions`): Your performance requirement is above an **upper confidence
#   bound** set based on label ambiguity.
# - **Insufficient evidence** (`not_ruled_out` or `unresolved`): The sweet-spot for model development, your requirements
#   lie in the "theoretically feasible and not yet achieved" window.
#
# An `evidence_conflict` might also occur, where the upper bound lies ~below~ the lower bound. This could indicate
# something wrong with the witness model (e.g. leakage artificially inflating performance),
# or an issue with the upper bound (e.g. incorrect assumptions when calculating label ambiguity). The
# [Performance limits](../concepts/PerformanceLimits.md#feasibility-ceilings-witnesses-and-determinations)
# concept page discusses this in more detail.
#
# Run `Feasibility` at two points: before model development, to learn whether a requirement is out of reach or
# already shown reachable, and after a model exists, to check that the model meets the requirement.
# This tutorial does both, using a model that has already been trained.
#
# `Feasibility` is experimental and may change in future releases.

# %% [markdown]
# ### The three performance requirements tested
#
# The program's main requirement is to achieve 60% accuracy when identifying a vehicle's name, but you have reason
# to believe that might be difficult due to visual similarity between vehicle name. So you plan to test two alternative
# paths to success as well. List all three performance targets before starting:
#
# 1. Vehicle name: achieve 60% accuracy when classifying the image as one of 24 different vehicle names.
# 2. Vehicle category: achieve 60% accuracy when classifying the image as one of 7 broader vehicle categories. Telling a
#    tank from an armored personnel carrier helps even when the exact model is unclear.
# 3. Tanks: achieve at least 90% recall, with at most a 5% false positive rate for the tank category.
#
# This tutorial tests all three performance targets using the same multi-class classifier (tuned once then frozen). The
# results that you will produce in this tutorial are summarized here:
#
# | Performance Target | Result | Justification |
# | --- | --- | --- |
# | Vehicle name is correct at least 60% of the time | Insufficient evidence | The witness model is correct about half the time, but the 60% target is not ruled out. |
# | Vehicle category is correct at least 60% of the time | Demonstrated | The witness model is right about 70% of the time, already meeting performance goals. |
# | Tank category has > 90% recall, < 5% false positives | Insufficient evidence | The witness model only achieves about 50% recall while meeting the false positive requirement, but 90% recall is not ruled out. |

# %% [markdown]
# ### Important Considerations
#
# #### Preventing multiple comparisons errors
#
# The evidence behind Feasibility's conclusions is based on statistical principals. When designing tests, practice good
# habits:
#
# - Assemble all the performance targets you plan on testing ahead of time so a cumulative 'error budget' can be
#   properly split between them
# - Conduct each set of tests on a different dataset (and no test set should be involved in any other part of the
#   process)
#
# Failing to follow these best practices can lead to inaccurate confidence bounds and misleading results.
#
# #### Choosing a witness model
#
# Ideally, `Feasibility` is run before a model has been trained, but in order to establish
# a relevant lower bound you will need to choose an appropriate witness model. This can be the same model you plan to
# use as a performance baseline in your future model evaluation, or any other existing model of the same type you are
# targeting (classification, object detection, etc.).
#
# This tutorial separates the witness model into an encoder (produces the `Embeddings` used in both the upper and lower
# `Feasibility` bounds) and a simple logistic-regression-based classifier which is trained on a subset of the data.
# This encoder has not seen your data, so its embeddings are likely not as well separated in feature space
# as your future trained model will be. Because the dataset in this tutorial only has one label per image, label
# ambiguity is based on groupings of images whose embeddings are close in feature space.
# This puts a limitation on the accuracy of the upper bound that is reflected in the `Feasibility` result printouts.

# %% [markdown]
# ## Setup
#
# Import the required libraries.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots maite-datasets
except Exception:
    pass

# %% tags=["remove_cell"]
import logging
import warnings

from dataeval.exceptions import ExperimentalWarning

# Feasibility and its evidence types are experimental; the tutorial says so once instead of warning in every cell.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# %%
import dataeval_plots as dep
import numpy as np
import torch
from maite_datasets.image_classification import MilitaryVehicles
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import radius_neighbors_graph
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ViT_B_16_Weights, vit_b_16

from dataeval import Embeddings, Metadata, Ontology
from dataeval.data import Indices, Relabel, View
from dataeval.extractors import TorchExtractor
from dataeval.performance import Feasibility, TargetSpec
from dataeval.quality import Duplicates

# %% tags=["remove_cell"]
# MilitaryVehicles downloads through huggingface_hub, which logs a warning for each rate-limited (HTTP 429)
# request it retries; maite-datasets already retries and reports progress. Set after the imports, because
# importing huggingface_hub resets its logger to WARNING.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# %% [markdown]
# ## Load the data
#
# `MilitaryVehicles`, from `maite-datasets`, has photos of 24 types of military vehicle in a training split and a test
# split. Each type belongs to one of 7 categories: Tank, BMP, BTR, Self Propelled Artillery, Air Defense, MT LB and BMD.

# %%
train = MilitaryVehicles(root="./data", image_set="train", download=True)
test = MilitaryVehicles(root="./data", image_set="test", download=True)
print(f"Training split: {len(train):,} images. Test split: {len(test):,} images.")
print(f"Vehicle names: {len(test.index2label)}")

# %% [markdown]
# ## Create an embedding for each image
#
# [Embeddings](../concepts/Embeddings.md) are used in multiple parts of this process - in your model, DataEval's
# built-in models, and the upper bound calculation. You use the 768-dimensional output from ViT-B/16
# (a `torchvision` model).

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ViT_B_16_Weights.DEFAULT
backbone = vit_b_16(weights=weights, progress=False)
backbone.heads = torch.nn.Sequential()  # an empty head keeps the 768-dimensional embedding, not a 1,000-class answer
extractor = TorchExtractor(backbone, transforms=weights.transforms(), device=device)

train_x = np.asarray(Embeddings(train, extractor=extractor, batch_size=128))
test_x = np.asarray(Embeddings(test, extractor=extractor, batch_size=128))
print(f"Each image is now a {train_x.shape[1]}-dimensional embedding")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (len(train), len(test), len(test.index2label)) == (7823, 1621, 24)
assert train_x.shape == (7823, 768)
assert test_x.shape == (1621, 768)

# %% [markdown]
# ## Remove duplicate images
#
# It is always good practice to remove duplicates from your datasets, especially when there is a possibility that some
# test images are duplicates of a training image to prevent 'memomrization' from inflating your measured test
# performance. Use DataEval's {class}`.Duplicates` class to find them.
#
# :::{dropdown} How the duplicates are found
# DataEval's {class}`.Duplicates` compares image fingerprints, which catch exact copies and resaved or resized ones.
# They miss some crops, padded borders, watermarks and mirror images, so two images also count as repeats when their
# embeddings have a **cosine similarity** (1 means identical) of at least 0.95. Check what that threshold catches before
# reusing it with another image model or dataset.
# :::

# %% tags=["hide-input"]
# DataEval's duplicate search, over both splits: each row is a group of repeats with matching fingerprints
found = Duplicates(hash_radius=0).evaluate(train, test).data()
pairs = []
for items, splits in zip(found["item_indices"], found["dataset_indices"], strict=True):
    ids = [item + split * len(train) for item, split in zip(items, splits, strict=True)]  # number training images first
    pairs += [(ids[0], other) for other in ids[1:]]

# Embeddings at cosine similarity >= 0.95, which for embeddings scaled to length 1 is a distance <= sqrt(2 - 2 * 0.95)
unit_embeddings = np.vstack([train_x, test_x])
unit_embeddings /= np.linalg.norm(unit_embeddings, axis=1, keepdims=True)
links = radius_neighbors_graph(unit_embeddings, radius=np.sqrt(2 - 2 * 0.95))
links = links + coo_matrix((np.ones(len(pairs)), tuple(zip(*pairs, strict=True))), shape=links.shape)

# A family is every image linked to another by either kind of repeat
_, family = connected_components(links, directed=False)
train_family, test_family = family[: len(train)], family[len(train) :]

repeats_training = np.isin(test_family, train_family)
first_of_family = np.unique(test_family, return_index=True)[1]
keep = np.sort(first_of_family[~repeats_training[first_of_family]])
repeats_test = len(test) - repeats_training.sum() - len(keep)
print(f"Removed {repeats_training.sum()} test images that repeat a training image")
print(f"Removed {repeats_test} test images that repeat another test image")
print(f"Kept {len(keep):,} test images")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Counts at the cosine-0.95 boundary move a little with the GPU, so they are checked in bands the prose stays true in
removed = len(test) - len(keep)
assert 0.08 <= removed / len(test) < 0.10  # "about 9% of the test split"
assert repeats_training.sum() > 2 * repeats_test  # "most of it because it repeats a training image"
assert 1460 <= len(keep) <= 1500  # "about 1,480"
assert len(np.unique(test_family[keep])) == len(keep)  # one image per family
assert not np.isin(test_family[keep], train_family).any()  # no family shared with training

# %% [markdown]
# ## Split your data
#
# | Dataset Split | Description | Job |
# | --- | --- | --- |
# | Train | About four fifths of the training split | Train your model. |
# | Calibration | About one fifth of the training split, reserved | Choose a classification threshold for the tank class (related to the third performance target). |
# | Test (all) | The reserved images, about 1,480 | Passed to `Feasibility`, which splits them in half at random. |
# | Test (used internally by DataEval) | Half of the test set, about 740 | Train DataEval's built-in models, and calculate label ambiguity precursors. |
# | Test (evaluation/audit) | The other half of the test set, about 740 | All the reported numbers come from these images. |
#
# `Feasibility` never sees a training image. Every claim below concerns the kept test images and new images from the
# same population.

# %% [markdown]
# ## Train your witness model
#
# Your witness model is a simple **logistic regression** classifier that learns, from the embeddings of the training
# images, a score between 0 and 1 for each of the 24 vehicle names, and predicts the highest scoring class.
# The model trains once then is **frozen**, meaning it is never changed again.

# %%
train_y = np.asarray(Metadata(train).class_labels)
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
fit_rows, calibration_rows = next(split.split(train_x, groups=train_family))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
model.fit(train_x[fit_rows], train_y[fit_rows])
scores = model.predict_proba(test_x)  # one row per test image, one score per vehicle name
print(f"Trained on {len(fit_rows):,} images; {len(calibration_rows):,} calibration images set aside")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.18 < len(calibration_rows) / len(train) < 0.22  # "about one fifth", by family
assert not np.isin(train_family[calibration_rows], train_family[fit_rows]).any()  # families stay on one side
assert model.classes_.tolist() == list(range(24))  # score columns are the vehicle names in index order

# %% [markdown]
# ## Splitting the error budget
#
# `alpha` is the error budget; the chance that any bound is wrong through bad luck in
# which images became evaluation images. The three `Feasibility` calls (one per performance target) share one 5% budget,
# allowing you to accept the group of results with a 95% confidence. You give each `Feasibility` call one third of the
# budget (`alpha = 0.05 / 3`) and use the same random `seed` so all three pick the same evaluation images.

# %%
feasibility = Feasibility(alpha=0.05 / 3, seed=42)  # for all three planned calls: a third of the budget each, one seed

# %% [markdown]
# ## Check the vehicle name requirement
#
# The call passes the kept test images, their embeddings, your model's scores and a record of how it was built:
#
# - `predictions` holds your model's 24 scores for each kept image, in image order. DataEval scores the model on the
#   evaluation images only.
# - `provenance` records how the model was built. `reads="representation"` declares that it reads only the embeddings.
#   DataEval records these declarations as assumptions and does not verify them.

# %%
types = View(test, [Indices(keep)])
provenance = {
    "reads": "representation",
    "frozen_before_audit": True,
    "selection": (
        "ViT-B/16 chosen by 5-fold cross-validation on the training split; logistic regression not tuned; an earlier "
        "exploratory comparison scored the three backbones on these test images"
    ),
    "preprocessing": "ViT-B/16 class-token embeddings, standardized",
}

type_result = feasibility.evaluate(
    types, target={"accuracy": 0.60}, embeddings=test_x[keep], predictions=scores[keep], provenance=provenance
)
print(type_result)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
NO_GAIN = "more evaluation images would not help at the observed scores;"  # quoted below
NEXT = "to show it, test a stronger model; to rule it out, collect repeated labels for a whole-image ceiling"
row = type_result.determinations.row(0, named=True)
assert (row["determination"], row["witness_source"], row["ceiling"]) == ("unresolved", "user_witness/v1", None)
assert row["suggested_minimum"] is None
printed = str(type_result)
flat = " ".join(printed.split())  # Next lines wrap after their semicolons
assert "All bounds hold together at 98.3% confidence" in printed
assert "They assume no evaluation image was used to train or choose a model, or to set a requirement." in printed
assert f"60% accuracy {NO_GAIN} {NEXT}" in flat
assert "accuracy) bounds only models that give one answer per group." in printed  # the group-based ceiling note
bounds = type_result.aggregate_by_evidence()
yours = bounds.filter(source="your model").row(0, named=True)
EVALUATION = len(keep) - round(len(keep) / 2)  # Feasibility fits on half the kept images and evaluates on the rest
assert yours["n"] == EVALUATION
assert 730 <= EVALUATION <= 750  # "about 740"
# A family count that moves at the cosine-0.95 boundary redraws the evaluation images; the bands cover such redraws
assert 0.44 < yours["observed"] < 0.56  # "about half"
assert 0.39 < row["witness"] < 0.48  # "roughly 40 to 45%"
group_based = bounds.filter(source="group-based ceiling")["bound"].item()
assert 0.51 < group_based < 0.60  # "about 55%"; below 60%, so it "would rule that out" at its scope

# %% [markdown]
# The answer is "insufficient evidence" (`unresolved` in the result's `determination` column). Your witness model named
# the vehicle correctly about half the time. Accounting for sampling, DataEval can say only that it gets at least
# roughly 40 to 45% right on new images. Because this dataset only has one ground truth label per image, the label
# ambiguity used to set the upper ceiling comes from grouping images of similar appearance. This group-based ceiling,
# about 55% accuracy, speaks only for models that would give one classification per "look-alike group", so it cannot
# firmly rule out your 60% accuracy performance target as "infeasible", but it can give a rough estimate.
#
# The "Next" section gives advice for what would provide a more concrete result. More evaluation images drawn from the
# same population would not help. Instead, address either the lower or upper bound:
#
# - Lower Bound: A stronger model that achieves 60% accuracy would demonstrate this performance target 'feasible'.
#   Either choose a more relevant witness model to use as a baseline, or proceed with training your own model.
# - Upper Bound: The preferred method of assessing label ambiguity is by assessing the variation between labels that
#   multiple different human annotators assign to the same image. Collect more labels per image to provide a better
#   estimate of the upper bound and potentially rule out the performance target as 'infeasible'.
#
# The "Notes" section states context and assumptions that are relevant for interpreting the results.
#
# Next, you assess the other two performance targets.

# %% [markdown]
# ## Check the vehicle category requirement
#
# This question uses the same images and the same frozen model, with coarser labels:
#
# - {class}`.Relabel` changes each image's label from its type to its category, using the dataset's own `hierarchy`,
#   read with {meth}`.Ontology.from_hierarchy`.
# - `relabel.fold_scores` does the same to your model's scores. It adds up the scores of the types in each category, so
#   the six tank types' scores become one Tank score. The model is not retrained.

# %%
ontology = Ontology.from_hierarchy(test.hierarchy)
categories = ontology.children("land vehicle")
to_category = {vehicle: category for category in categories for vehicle in ontology.subtree_ids(category)}
relabel = Relabel(to_category, target=categories)

by_category = View(test, [Indices(keep), relabel])
category_scores = relabel.fold_scores(scores, view=by_category)  # one row per kept image, one score per category

category_result = feasibility.evaluate(
    by_category, target={"accuracy": 0.60}, embeddings=test_x[keep], predictions=category_scores, provenance=provenance
)
print(category_result)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert categories == ("Tank", "BMP", "BTR", "Self Propelled Artillery", "Air Defense", "MT LB", "BMD")
assert sum(to_category[name] == "Tank" for name in test.index2label.values()) == 6  # "the six tank types"
assert category_scores.shape == (len(keep), 7)
row = category_result.determinations.row(0, named=True)
assert (row["determination"], row["witness_source"]) == ("demonstrated_feasible", "user_witness/v1")
assert 0.62 <= row["witness"] < 0.69  # "at least about 65%"
yours = category_result.aggregate_by_evidence().filter(source="your model").row(0, named=True)
assert yours["n"] == EVALUATION
assert 0.65 < yours["observed"] < 0.74  # "about 70%"

# %% [markdown]
# The performance target is "demonstrated" (`demonstrated_feasible`). Your witness model achieved about 70% accuracy.
# Allowing for sampling, the lower bound on feasibility is set to ~65%, which exceeds the 60% target.
#
# `dataeval_plots` provides a helpful visualization of the results. In the chart below, the blue region represents the
# demonstrated performance, showing the lower bound on feasibility as a vertical blue rule. The gray zone is the set of
# accuracies that have neither been demonstrated or ruled out by the upper bound. Your performance target is marked by
# the dashed line.

# %%
dep.plot(category_result)

# %% [markdown]
# ## Check the tank-discrimination requirement
#
# You are concerned with the data's ambiguity regarding tanks in particular. You
# recall that so far, your model has been assigning the class of an image based on which class was 'most probable', but
# for this problem, you decide that the model should call an image a tank only when the probability of it being a tank
# exceeds a certain **threshold** instead. Below the threshold, the next highest scoring class is chosen instead.
# Increasing the threshold will result in fewer false positives and more missed tanks. Being the good,
# laws-of-statistics-abiding scientist you are, you chose this threshold using the held-out calibration data.
#
# Your performance target states that the model must be able to find at least 90% of tanks (recall) while calling at
# most 5% of the other vehicles tanks (false positives).
#
# :::{dropdown} Choosing a threshold
# You want your false positive rate to be no more than 5%, but there are many sources of uncertainty, so you chose
# the lowest threshold that called at most 2.5% of other vehicles tanks - a slightly stricter rule. The first source of
# uncertainty is in the construction of the "no more than" number for false alarms. By design, that number sits slightly
# above the share of false positives actually observed. The share of false positives observed for the calibration data
# will likely also differ from the share of false positives observed for the test set by chance. The stricter threshold
# gives a better chance of seeing the performance target met using this feasibility test.
# :::

# %% tags=["hide-input"]
TANK = categories.index("Tank")
assert train.index2label == test.index2label  # both splits number the 24 types the same way
category_of_type = np.array([categories.index(to_category[name]) for name in train.index2label.values()])
is_tank = category_of_type[train_y[calibration_rows]] == TANK
calibration_scores = relabel.fold_scores(model.predict_proba(train_x[calibration_rows]))
tank_score = calibration_scores[:, TANK]

# The declared rule: the lowest threshold that calls at most 2.5% of the calibration non-tanks tanks
candidates = np.unique(tank_score)
false_alarms = (tank_score[~is_tank] >= candidates[:, None]).mean(axis=1)
threshold = candidates[np.argmax(false_alarms <= 0.025)]


def report(says_tank, rule):
    """Print the share of calibration tanks a rule finds, and of other vehicles it calls tanks."""
    print(f"{rule}: finds {says_tank[is_tank].mean():.1%} of the calibration tanks, ", end="")
    print(f"calls {says_tank[~is_tank].mean():.1%} of the other vehicles tanks")


report(calibration_scores.argmax(axis=1) == TANK, "Highest score wins")
report(tank_score >= threshold, f"Tank score >= {threshold:.4f}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
highest = calibration_scores.argmax(axis=1) == TANK
# A family count that moves at the cosine-0.95 boundary redraws the calibration split; the bands cover such redraws
assert 0.74 < highest[is_tank].mean() < 0.86  # "about 80% of the tanks"
assert 0.07 < highest[~is_tank].mean() < 0.12  # "about 10% of the other vehicles"
assert 0.02 < false_alarms[candidates == threshold].item() <= 0.025  # "2.5% or just under"
assert false_alarms[candidates < threshold].min() > 0.025  # the lowest threshold that meets the rule
assert 0.40 < (tank_score[is_tank] >= threshold).mean() < 0.60  # "only about half of the tanks"

# %% [markdown]
# On the calibration images, using the highest-scoring class finds about 80% of the tanks but calls about 10% of the
# other vehicles tanks, well over the 5% limit. The chosen threshold brings the false positives down to 2.5% or just
# under.
#
# Now you apply this rule to the `Feasibility` test. `predictions` takes those decisions, one category number per image,
# and `provenance["decision_policy"]` records the decision threshold. Testing an additional threshold would need a
# planned call of its own (and a new test set, as per best practices for statistical testing).


# %% tags=["hide-input"]
def decide(folded):
    """Tank when the Tank score reaches the threshold; otherwise the highest-scoring other category."""
    others = folded.copy()
    others[:, TANK] = -np.inf
    return np.where(folded[:, TANK] >= threshold, TANK, others.argmax(axis=1))


tank_requirement = TargetSpec(
    task="IC", metric="recall", threshold=0.90, class_name="Tank", max_false_positive_rate=0.05
)
tank_provenance = provenance | {
    "decision_policy": (
        f"Tank (index {TANK}) when the summed Tank score is at least {threshold}, otherwise the highest-scoring "
        "other category; the threshold is the lowest whose false-positive rate on a calibration fifth of the "
        "training split, held out by near-duplicate family, is at most 2.5%"
    )
}

tank_result = feasibility.evaluate(
    by_category,
    target=[tank_requirement],
    embeddings=test_x[keep],
    predictions=decide(category_scores),
    provenance=tank_provenance,
)
print(tank_result)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
row = tank_result.determinations.row(0, named=True)
assert row["determination"] == "unresolved"
assert (row["ceiling"], row["suggested_minimum"]) == (None, None)
printed = str(tank_result)
assert "Tank: ≥90% recall, ≤5% false positives   insufficient evidence ?" in printed
TANK_NEXT = NEXT.replace("model;", "model or a different frozen operating point;")
assert f"Tank {NO_GAIN} {TANK_NEXT}" in " ".join(printed.split())
yours = tank_result.aggregate_by_evidence().filter(source="your model")
assert yours["quantity"].to_list() == ["recall", "false_positive_rate"]
tanks, others = yours["n"].to_list()
assert tanks + others == EVALUATION
assert 560 <= others <= 640  # "about 600 other vehicles"
assert 0.38 < yours["observed"][0] < 0.62  # "only about half of the tanks"
assert 0.01 < yours["observed"][1] < 0.05  # "a few percent"
# "close to the 5% limit": a redraw of the evaluation images can land it on either side, which the prose covers
assert 0.03 < yours["bound"][1] < 0.08
counted = row["witness_source"] == "user_witness/v1"
assert counted == (yours["bound"][1] <= 0.05)  # counted when false alarms are shown within the limit
assert "FP ≤" in printed  # "shows both of the rule's numbers either way"
assert ("(your model, not counted)" in printed) != counted  # "above 5%, it marks them 'not counted'"
user = tank_result.evidence["user_witness/v1"]
assert user.provenance["decision_policy"] == tank_provenance["decision_policy"]
# the three planned calls: one budget split three ways, the same seed and the same evaluation images
for result in (type_result, category_result, tank_result):
    assert result.evidence["user_witness/v1"].details["call_alpha"] == 0.05 / 3
    assert result.evidence["user_witness/v1"].details["seed"] == 42
    assert result.determinations["suggested_minimum"].to_list() == [None]

# %% [markdown]
# The result shows "insufficient evidence" (`unresolved`). The new tank classification rule had a significantly lower
# recall on tank images, far below 90%, so the witness model has not demonstrated the performance target.

# %% [markdown]
# ## Choose your next steps
#
# When running `Feasibility`, there are 3 categories of results you can expect:
#
# 1. The target performance is higher than the upper bound: Given your data, your requirement is not achievable.
#    Either re-evaluate your performance goals, or (more likely) collect new, **different** data that more clearly
#    distinguishes between your classes.
# 2. The requirement is below the lower bound: Your witness model is already able to achieve your desired
#    performance, and you do not need to invest heavily in model training or development.
# 3. The requirement lies between the demonstrated bounds: Your target performance is feasible, but has not yet been
#    demonstrated by an existing model, so you can proceed with model training and development, reasonably confident
#    that **the data is not a limiting factor** for performance.
#
# ### Interpreting the specific recommendations provided by `Feasibility`
#
# - **Test a stronger model, to show a requirement met.** Only a tested model's "at least this" performance number can
#   show a requirement met, so you can try out a more relevant model if you want to ensure that your performance target
#   is not yet reachable using existing tools, but you might instead want to proceed with training your own model
#   (provided that the target performance lies below the upper bound).
# - **Collect repeated labels, to rule a requirement out.** Have several people label the same random subset of the
#   images independently, and pass their labels as `replicas=`. Where the labelers disagree about one image, it implies
#   inherent ambiguity between classes for the given data, so the disagreement sets a limit on how well any model can
#   do. That limit is a better estimate of the upper performance bound than the group-based approach used in the absence
#   of `replicas`.
# - **Assert allowances, to rule a requirement out.** setting `allowances={"beta": ..., "gamma": ...}` allows DataEval
#   to use the group-based performance ceiling as a proper upper bound. These values will be guesses. `beta` estimates
#   how much better you expect a model to be able to do when not limited by the coarse appearance-wise groupings used to
#   evaluate the label ambiguity bound. `gamma` estimates how much better your fully trained embeddings will be compared
#   to the witness model's embeddings. DataEval cannot check either number, so the ruling-out holds only if your
#   assertion does.
# - **Ask a narrower question.** If your performance targets are unreasonable, the solution may involve re-evaluating
#   the problem and assessing new performance targets. When changing the target, ensure you follow the best practices
#   listed above.
#
# When a model's observed score already meets the requirement and only the allowance for sampling holds it back,
# `suggested_minimum` is set and "Next" gives about how many evaluation images would carry its bound past the
# requirement. `Feasibility` evaluates on half of the images you pass, so supply about twice the suggested minimum.

# %% [markdown]
# ## Inspect the evidence
#
# This section is optional. `aggregate_by_evidence` lists every bound of a `Feasibility` call, one row per source and
# bound. For the tank-specific recall/false positive example, each tested model has two rows: a lower bound on the share
# of tanks it finds (`recall`) and an upper bound on the share of other vehicles it calls tanks (`false_positive_rate`).
# The `information_scope` column gives the kind of model each bound applies to. "Not counted" refers to a tested model
# whose false positives are not shown within the limit.

# %%
columns = ["source", "quantity", "bound_type", "bound", "observed", "n", "information_scope"]
tank_result.aggregate_by_evidence().select(columns)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
by_source = {(r["source"], r["quantity"]): r for r in tank_result.aggregate_by_evidence().iter_rows(named=True)}
assert 0.66 < by_source["built-in k-NN", "recall"]["observed"] < 0.84  # "about three quarters"
assert 0.66 < by_source["built-in linear probe", "recall"]["observed"] < 0.84
assert 0.24 < by_source["built-in k-NN", "false_positive_rate"]["bound"] < 0.32  # "25 to 30%", not counted
assert 0.12 < by_source["built-in linear probe", "false_positive_rate"]["bound"] < 0.19  # "about 15%", not counted
assert by_source["group-based ceiling", "recall"]["information_scope"] == "partition_restricted"

# %% [markdown]
# `observed` is a model's score on the `n` evaluation images. DataEval's two built-in models answer with the
# highest-scoring category. They find about three quarters of the tanks, but their false positive rates could be as high
# as 25 to 30% (k-NN) and about 15% (linear probe), so neither can claim the performance target reached.
# [Feasibility evidence](../concepts/FeasibilityEvidence.md) explains how each bound is built.
#
# In the evidence chart, each row is one source. The dot is its score on the evaluation images and the tick its "at
# least" number. Sources that are not counted, and the group-based ceiling, are drawn faint.

# %%
dep.plot(tank_result, plot_evidence=True)

# %% [markdown]
# ## Conclusion
#
# Three questions were asked of `Feasibility`:
#
# 1. When classifying a vehicle by **name**, is my > 60% accuracy requirement feasible given this data?
# 2. When classifying a vehicle by **category**, is my > 60% accuracy requirement feasible given this data?
# 3. When classifying a vehicle by category and using a **stricter decision rule** for tanks, is my requirement
#    of > 90% recall and < 5% false positive rate feasible given this data?
#
# Only the second performance goal was demonstrated by the witness model, meaning the other two goals will benefit from
# training a new, stronger model. The third performance goal was shown to lie below the upper performance bound
# (feasible). The first performance goal, however, rests just above the reported upper bound. Because
# the upper bound was determined using the look-alike group method, you decide collect multiple label replicates per
# image to get a better estimate of the performance ceiling before committing to training a new model.
#
# The answers to the three questions asked of `Feasibility` hold together at 95% confidence for new images from the same
# population, and rest on the context declared in the model `provenance`.

# %% [markdown]
# ## Next steps
#
# - [Performance limits](../concepts/PerformanceLimits.md) — Understand the determinations, information scopes and
#   recommendations of `Feasibility`.
# - [Feasibility evidence](../concepts/FeasibilityEvidence.md) — Learn how each kind of evidence is built and what its
#   details hold.
# - [Measure dataset sufficiency for image classification](./h2_measure_ic_sufficiency.py) — Project model
#   performance curves as a function of dataset size.

# %% [markdown]
# ## On your own
#
# Repeat the assessment on your own classification dataset:
#
# - Write down every requirement before you run anything, and give them one error budget with `alpha`.
# - Create embeddings for your images with a pretrained model, and pass them to `Feasibility.evaluate` with your
#   dataset.
# - Train/tune a witness model on images that `Feasibility` never sees, freeze it, and pass its class scores as
#   `predictions=` with a `provenance` that records how you built it.
# - Remove any duplicates between your training and evaluation images, as in the "Remove repeated images" section.
# - Use evaluation images that nobody has looked at, so the stated confidence holds true.
# - If a requirement stays undecided, follow the "Choose your next steps" section.
