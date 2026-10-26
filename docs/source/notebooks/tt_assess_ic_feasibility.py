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
# This tutorial uses {class}`~dataeval.performance.Feasibility` to decide whether three requirements for a military
# vehicle classifier are reachable on the `MilitaryVehicles` images.
#
# Estimated time to complete: 30 minutes
#
# Relevant ML stages: [Model Development](../getting-started/roles/ML_Lifecycle.md#model-development)
#
# Relevant personas: [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you will do
#
# - You will ask {class}`~dataeval.performance.Feasibility` whether a model can name the type of military vehicle in a
#   photo well enough, using the `MilitaryVehicles` images: 24 vehicle types in 7 categories.
# - When the evidence cannot answer that, you will change the question twice: to the vehicle's category, then to tanks
#   alone.
# - You will read each answer as a printout and as a chart.

# %% [markdown]
# ## What you will learn
#
# - You will learn why only a tested model's "at least this" number can show a requirement met, while a "no more than
#   this" number can only rule one out.
# - You will learn how to change a question the evidence cannot answer, and when that change is fair.
# - You will learn how to ask a coarser question of the same model by relabeling the images and adding up its scores.
# - You will learn how to choose a tank decision rule on training images and check it once on held-out images.

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
# - A GPU is optional; it makes the image summaries much faster to compute

# %% [markdown]
# ## Introduction
#
# Your program is building a system that names the military vehicle in a photo, and wants to know whether its
# requirement is reachable before paying for model development. {class}`~dataeval.performance.Feasibility` answers for
# each requirement. The evidence shows it met (**demonstrated**), shows it cannot be met under the assumptions it
# records (**ruled out**), or cannot decide yet (**insufficient evidence**). The decision stays yours. `Feasibility` is
# experimental and may change in future releases.
#
# When the evidence cannot decide, you can **change the question**: ask a different, clearly stated one that the mission
# would accept, and report the answer as an answer to that question.

# %% [markdown]
# ### Preview the three answers
#
# The sections after the setup show how the tutorial reaches these answers.
#
# | Question | Answer | Why |
# | --- | --- | --- |
# | Vehicle type: right for at least 60% of new images? | Insufficient evidence | Your model is right about half the time, so it cannot show 60%, and more evaluation images would not change that. Nothing rules 60% out. |
# | Vehicle category: right for at least 60% of new images? | Demonstrated | Your model is right about 70% of the time, and at least about 65% allowing for sampling. |
# | Tanks: find at least 90% of them, with at most 5% false alarms? | Insufficient evidence | Your tank rule finds only about half of the tanks. Nothing rules 90% out. |
#
# "Insufficient evidence" describes the requirement, not the model: the evidence neither shows the requirement met nor
# rules it out.
#
# These answers illustrate the procedure and do not confirm the requirements. The stated confidence assumes that no
# evaluation image was used to train or choose a model, or to set a requirement, and this tutorial tried its thresholds
# and tank rule on these test images first. A formal check needs evaluation images nobody has looked at.

# %% [markdown]
# ### Read the printout
#
# DataEval reports two kinds of bounds. Each holds at the stated confidence: at 95%, the chance that any bound in the
# evaluation is wrong is at most 5%.
#
# - The **lower confidence bound** comes from a model you tested: the model's accuracy on new images from the same
#   population is at least this number. It sits below the accuracy observed on the evaluation images, to allow for
#   sampling.
# - The **upper confidence bound** comes from a **ceiling**: no model of the stated kind exceeds this accuracy on new
#   images from the same population, under the recorded assumptions.
#
# **New images from the same population** means images like these, gathered the same way. **Allowing for sampling**
# means allowing for luck in which images were drawn. A ceiling comes from the labels: where look-alike images carry
# different labels, a model that must answer them alike cannot get them all right. For false alarms, less is better, so
# a tested model gets a "no more than" number for them.
#
# Each printout is a table with one row per requirement, then "Next" and "Notes":
#
# | In the printout | In plain words |
# | --- | --- |
# | Tank: ≥90% recall, ≤5% false positives | Finds at least 90% of the tanks (**recall**) while calling at most 5% of the other vehicles tanks (**false positives**, or false alarms). |
# | demonstrated ✓ | A model you tested is shown to meet the requirement. |
# | insufficient evidence ? | The evidence cannot decide either way yet. |
# | models that may use the whole image | The kind of model a bound speaks for, its **scope**. Every requirement here is judged for these. |
# | group-based ceiling | An upper bound for models that give one answer per group of look-alike images; it says nothing about a model that tells them apart. |
# | not counted | A tested model whose false alarms are not shown within the limit. Its numbers cannot show the requirement met. |
# | All bounds hold together at 98.3% confidence | The chance that any bound in this printout is wrong is at most about 1.7%. |
# | They assume no evaluation image was used… | The confidence holds only if the evaluation images played no part in building a model or setting a requirement. |
# | evaluation images | The images DataEval holds back to score every model once (the **audit**). |
# | Next, Notes | What could settle an undecided requirement; the models' observed scores and narrower bounds. |

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
# `MilitaryVehicles`, from `maite-datasets`, has photos of 24 types of military vehicle, such as the T-72 tank or the
# BTR-80 armored personnel carrier, in a training split and a test split. Each type belongs to one of 7 categories:
# Tank, BMP, BTR, Self Propelled Artillery, Air Defense, MT LB and BMD.

# %%
train = MilitaryVehicles(root="./data", image_set="train", download=True)
test = MilitaryVehicles(root="./data", image_set="test", download=True)
print(f"Training split: {len(train):,} images. Test split: {len(test):,} images.")
print(f"Vehicle types: {len(test.index2label)}")

# %% [markdown]
# ## Summarize each image as numbers
#
# Your model, DataEval's built-in models and the search for repeated images all read
# [embeddings](../concepts/Embeddings.md): for each image, a list of numbers that a pretrained image model computes to
# summarize what the image shows. Similar-looking images get similar lists. You use ViT-B/16, from `torchvision`, and
# keep its 768 numbers.
#
# DataEval assumes the image model was chosen without the evaluation images' labels. ViT-B/16 was chosen over ResNet18
# and ResNet50 by **cross-validation** on the training split: train on four fifths, score on the last fifth, five times
# over. An earlier, exploratory comparison also looked at the test images.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ViT_B_16_Weights.DEFAULT
backbone = vit_b_16(weights=weights, progress=False)
backbone.heads = torch.nn.Sequential()  # an empty head keeps the 768-number summary, not a 1,000-class answer
extractor = TorchExtractor(backbone, transforms=weights.transforms(), device=device)

train_x = np.asarray(Embeddings(train, extractor=extractor, batch_size=128))
test_x = np.asarray(Embeddings(test, extractor=extractor, batch_size=128))
print(f"Each image is now {train_x.shape[1]} numbers")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (len(train), len(test), len(test.index2label)) == (7823, 1621, 24)
assert train_x.shape == (7823, 768)
assert test_x.shape == (1621, 768)

# %% [markdown]
# ## Remove repeated images
#
# Some test images repeat a training image: copies, crops, padded or watermarked versions and mirror images. A model can
# score well on those by remembering, which makes it look better than it is. DataEval also treats every evaluation image
# as a separate example, which two copies of one photo are not. So the images are sorted into **families** of repeats,
# every test image whose family includes a training image is removed, and one image per remaining family is kept.
#
# :::{dropdown} How the repeats are found
# DataEval's {class}`.Duplicates` compares image fingerprints, which catch exact copies and resaved or resized ones. They
# miss some crops, padded borders, watermarks and mirror images, so two images also count as repeats when their
# summaries have a **cosine similarity** (1 means identical) of at least 0.95. Check what that threshold catches before
# reusing it with another image model or dataset.
# :::

# %% tags=["hide-input"]
# DataEval's duplicate search, over both splits: each row is a group of repeats with matching fingerprints
found = Duplicates(hash_radius=0).evaluate(train, test).data()
pairs = []
for items, splits in zip(found["item_indices"], found["dataset_indices"], strict=True):
    ids = [item + split * len(train) for item, split in zip(items, splits, strict=True)]  # number training images first
    pairs += [(ids[0], other) for other in ids[1:]]

# Summaries at cosine similarity >= 0.95, which for summaries scaled to length 1 is a distance <= sqrt(2 - 2 * 0.95)
summaries = np.vstack([train_x, test_x])
summaries /= np.linalg.norm(summaries, axis=1, keepdims=True)
links = radius_neighbors_graph(summaries, radius=np.sqrt(2 - 2 * 0.95))
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
# That removes about 9% of the test split, most of it because it repeats a training image. A repeat that neither method
# catches would still make the scores look better than they are.

# %% [markdown]
# ## Train your model
#
# Your model is a **logistic regression**: a simple classifier that learns, from the training images' summaries, a score
# between 0 and 1 for each of the 24 vehicle types, and answers with the highest. About one fifth of the training
# images, the **calibration** images, are set aside for choosing the tank rule later, with each family of repeats kept
# on one side. The model trains once on the rest and is then **frozen**: it answers all three questions unchanged.

# %%
train_y = np.asarray(Metadata(train).class_labels)
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
fit_rows, calibration_rows = next(split.split(train_x, groups=train_family))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
model.fit(train_x[fit_rows], train_y[fit_rows])
scores = model.predict_proba(test_x)  # one row per test image, one score per vehicle type
print(f"Trained on {len(fit_rows):,} images; {len(calibration_rows):,} calibration images set aside")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.18 < len(calibration_rows) / len(train) < 0.22  # "about one fifth", by family
assert not np.isin(train_family[calibration_rows], train_family[fit_rows]).any()  # families stay on one side
assert model.classes_.tolist() == list(range(24))  # score columns are the vehicle types in index order

# %% [markdown]
# ## Assign a job to each set of images
#
# | Images | Job |
# | --- | --- |
# | About four fifths of the training split | Train your model. |
# | About one fifth of the training split, the calibration images | Choose the tank decision rule, in the "Check the tank requirement" section. |
# | The kept test images, about 1,480 | Passed to `Feasibility`, which splits them in half at random. |
# | Half of those | Train DataEval's built-in models, and sort the images into groups of similar-looking images. |
# | The other half, about 740, the *evaluation images* | Score every model, once. All the reported numbers come from these images. |
#
# `Feasibility` never sees a training image. Every claim below concerns the kept test images (those in which neither
# method found a repeat of a training image, one per family) and new images from the same population.

# %% [markdown]
# ## Plan the questions before you look
#
# The program's requirement is to name the vehicle type correctly for at least 60% of new images. With 24 look-alike
# types the evidence may not show that. Before you run anything, write down every requirement you will ask, including
# two that the mission would accept:
#
# - Vehicle type: name the right type, out of 24, for at least 60% of new images.
# - Vehicle category: name the right category, out of 7, for at least 60% of new images. Telling a tank from an armored
#   personnel carrier helps even when the exact model is unclear.
# - Tanks: find at least 90% of the tanks, while calling at most 5% of the other vehicles tanks.
#
# Each has its own answer. An answer to the category or tank requirement says nothing about the vehicle type
# requirement.
#
# `alpha` caps the chance that a call's bounds are wrong through bad luck in which images became evaluation images. The
# three calls share one 5% budget: each runs at `alpha = 0.05 / 3` with the same `seed`, so all three pick the same
# evaluation images and hold together at 95% confidence. Each printout states its own 98.3%, rounded down so that it
# never overstates.
#
# Changing the question is fair when the new question is one the mission would accept, is written down before any result
# is seen, shares the one budget, and has its answer reported as an answer to that question. Adding questions or moving
# thresholds after seeing results is not fair: with enough tries, some question passes by luck. This tutorial chose its
# thresholds after trying them on these test images, so its results only illustrate the procedure.

# %%
feasibility = Feasibility(alpha=0.05 / 3, seed=42)  # for all three planned calls: a third of the budget each, one seed

# %% [markdown]
# ## Check the vehicle type requirement
#
# The call passes the kept test images, their summaries, your model's scores and a record of how it was built:
#
# - `predictions` holds your model's 24 scores for each kept image, in image order. DataEval scores the model on the
#   evaluation images only.
# - `provenance` records how the model was built; `reads="representation"` declares that it reads only the summaries.
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
# The answer is "insufficient evidence". Your model named the right type for about half of the evaluation images.
# Allowing for sampling, DataEval can say only that it gets at least roughly 40 to 45% right on new images. Nothing
# rules 60% out either: the group-based ceiling, about 55%, speaks only for models that give one answer per group, and
# your model tells individual images apart.
#
# "Next" says what could settle it. More evaluation images would not help, because the observed score is already below
# 60%. A stronger model could show 60% met; repeated labels could rule it out. The printout's third line states the
# assumption behind its confidence, which this tutorial does not fully meet. Next, you ask the other questions you
# planned.

# %% [markdown]
# ## Check the vehicle category requirement
#
# This question uses the same images and the same frozen model, with coarser labels:
#
# - {class}`.Relabel` changes each image's label from its type to its category, using the dataset's own `hierarchy`,
#   read with {meth}`.Ontology.from_hierarchy`.
# - `relabel.fold_scores` does the same to your model's scores: it adds up the scores of the types in each category, so
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
# The answer is "demonstrated". Your model named the right category for about 70% of the evaluation images. Allowing for
# sampling, it gets at least about 65% right on new images, which meets 60%. This rests on your declared provenance: the
# model was trained without the evaluation images and frozen before scoring them. Removing the repeats that were found
# supports the first part.
#
# In the chart, the blue zone holds the requirements the evidence demonstrates, up to your model's "at least" number;
# the gray zone holds those it cannot decide. The marker is your requirement.

# %%
dep.plot(category_result)

# %% [markdown]
# ## Check the tank requirement
#
# The tank requirement has two numbers, and both must hold for one way of deciding: find at least 90% of the tanks
# (recall) while calling at most 5% of the other vehicles tanks (false positives).
#
# Your model's usual answer is the category with the highest score. To call fewer vehicles tanks, the system can answer
# "Tank" only when the Tank score reaches a **threshold**, and otherwise give the highest-scoring other category. A
# higher threshold means fewer false alarms and more missed tanks. Choosing it takes four steps:
#
# 1. State the requirement first, as above.
# 2. Choose the threshold on training images only, by a rule declared before looking: on the calibration images, the
#    lowest threshold that calls at most 2.5% of the other vehicles tanks.
# 3. Freeze the model, the scores and the threshold into one decision per test image.
# 4. Check once on the evaluation images, in a call of its own.
#
# :::{dropdown} Why 2.5% and not 5%
# The evaluation images will give a "no more than" number for false alarms a few points above the share they show, and
# that share can itself come out above the calibration share. A rule that sat near 5% on the calibration images would
# leave no room.
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
# On the calibration images, the highest-score answer finds about 80% of the tanks but calls about 10% of the other
# vehicles tanks, well over the 5% limit. The threshold brings the false alarms down to 2.5% or just under, and finds
# only about half of the tanks. The highest-score numbers say nothing about whether some threshold could meet the
# requirement, because the two are different rules.
#
# Now the frozen rule decides each test image. `predictions` takes those decisions, one category number per image, and
# `provenance["decision_policy"]` records the rule. A different threshold would need a planned call of its own.


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
assert counted == (yours["bound"][1] <= 0.05)  # counted exactly when false alarms are shown within the limit
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
# The answer is "insufficient evidence". The rule found only about half of the tanks among the evaluation images, far
# below 90%, so it cannot show the requirement met whatever its false alarms. "Next" says so first: more evaluation
# images would not help at the observed scores.
#
# The evaluation images hold about 600 other vehicles. The rule called a few percent of them tanks, and DataEval's "no
# more than" number for that share comes out close to the 5% limit. The "Lower confidence bound" column shows both of
# the rule's numbers either way, "recall" and "FP ≤"; above 5%, it marks them "not counted", and the Notes say why. A
# lower threshold that found more tanks would usually also call more of the other vehicles tanks.

# %% [markdown]
# ## Choose what to do next
#
# The `recommendation` column of each result's `determinations` details the "Next" line's routes:
#
# - To show a requirement met, test a stronger model. Only a tested model's "at least" number can show it met, so you
#   need one that scores well above the requirement on the evaluation images (for the tanks, with false alarms shown
#   within 5%). These calls spent the whole 5% budget, so judge it on fresh evaluation images.
# - To rule a requirement out, collect repeated labels. Have several people label a random subset of the images
#   independently and pass their labels as `replicas=`. Where they disagree, no single answer matches every label, so
#   the labels give a ceiling for models that may use the whole image.
# - To rule a requirement out, assert allowances. `allowances={"beta": ..., "gamma": ...}` extend the group-based
#   ceiling on accuracy to models that may use the whole image. An allowance is a bound you assert and DataEval cannot
#   check; see [what an allowance asserts](../concepts/FeasibilityEvidence.md#allowances).
# - Ask a narrower question. The key `"accuracy/partition_restricted"` asks whether a model that gives one answer per
#   group can reach 60%. The group-based ceiling, about 55%, would rule that out under the recorded assumptions, but
#   says nothing about a model like yours. It too would need planning and its own share of the budget.
#
# When a model's observed score already meets the requirement and only the allowance for sampling holds it back,
# `suggested_minimum` is set and "Next" gives about how many evaluation images would carry its bound past the
# requirement. `Feasibility` evaluates on half of the images you pass, so supply about twice that many.

# %% [markdown]
# ## Inspect the evidence
#
# This section is optional. `aggregate_by_evidence` lists every bound of a call, one row per source and bound. For the
# tank requirement, each tested model has two rows: a lower bound on the share of tanks it finds (`recall`) and an upper
# bound on the share of other vehicles it calls tanks (`false_positive_rate`).

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
# highest-scoring category. They find about three quarters of the tanks, but their false alarms could be as high as 25
# to 30% (k-NN) and about 15% (linear probe), so neither counts.
# [Feasibility Evidence](../concepts/FeasibilityEvidence.md) explains how each bound is built.
#
# In the evidence chart, each row is one source: the dot is its score on the evaluation images and the tick its "at
# least" number. Sources that are not counted, and the group-based ceiling, are drawn faint.

# %%
dep.plot(tank_result, plot_evidence=True)

# %% [markdown]
# ## Conclusion
#
# The vehicle type at 60% could not be answered. Changing the question gave one requirement the evidence demonstrates,
# the vehicle category, and one whose shortfall says what to do next, the tanks. The three answers hold together at 95%
# confidence for new images from the same population, and they rest on your model's declared provenance. Because this
# tutorial tried its thresholds and rules on these test images, they illustrate the procedure. A formal check needs
# fresh evaluation images.

# %% [markdown]
# ## Next steps
#
# - [Performance Limits](../concepts/PerformanceLimits.md) — Understand the determinations, information scopes and recommendations of `Feasibility`.
# - [Feasibility Evidence](../concepts/FeasibilityEvidence.md) — Learn how each kind of evidence is built and what its details hold.
# - [How to determine image classification feasibility](./h2_measure_ic_feasibility.py) — Estimate the Bayes error rate with `ber_mst` and check an accuracy requirement against it.
# - [How to measure dataset sufficiency for image classification](./h2_measure_ic_sufficiency.py) — Project model performance curves as a function of dataset size.

# %% [markdown]
# ## On your own
#
# Repeat the assessment on your own classification dataset:
#
# - Write down every requirement before you run anything, and give them one error budget with `alpha`.
# - Compute embeddings for your images with a pretrained model, and pass them to `Feasibility.evaluate` with your
#   dataset.
# - Train your model on images that `Feasibility` never sees, freeze it, and pass its class scores as `predictions=`
#   with a `provenance` that records how you built it.
# - Remove the repeats between your training and evaluation images, as in the "Remove repeated images" section.
# - Use evaluation images that nobody has looked at, so the stated confidence holds.
# - If a requirement stays undecided, follow the "Choose what to do next" section.
