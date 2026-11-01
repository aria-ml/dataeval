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
# # Identify out-of-distribution samples
#
# Learn how to identify [out-of-distribution](../concepts/DistributionShift.md#out-of-distribution-detection) (OOD)
# samples using reconstruction-based methods with different model architectures, and distance-based methods.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Monitoring](../getting-started/roles/ML_Lifecycle.md#monitoring),
# [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [ML Engineer](../getting-started/roles/ml_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md), [Data Scientist](../getting-started/roles/data_scientist.md)
#
# ## What you will do
#
# - Fit six detectors on photographs of ground vehicles: KNN with cosine and with Euclidean distance on pretrained
#   embeddings, and reconstruction with an AE, a VAE, and each of those with a Gaussian mixture model (GMM) head
# - Score four kinds of input that the detectors never saw: noisy frames, blurred frames, aerial maritime frames, and a
#   vehicle family that was left out of training
# - Compare which detector catches which kind of input
# - Read OOD scores against a percentile threshold, and visualize reconstructions
#
# ## What you will learn
#
# - How to fit the six detectors on in-distribution data and score new data with each
# - That a reconstruction detector flags inputs that are hard to reconstruct, such as noise, and misses inputs that are
#   easy to reconstruct, such as blurred or aerial frames
# - That distance between pretrained embeddings catches a change of sensor or platform, and that no detector here
#   catches a new vehicle of a known kind
# - How to read OOD scores against a percentile threshold
#
# ## What you will need
#
# - Knowledge of Python
# - Basic understanding of PyTorch and neural networks
# - Understanding of autoencoders (helpful but not required)

# %% [markdown]
# ## Introduction
#
# Out-of-distribution (OOD) detection helps ensure model reliability in production. When models encounter data that
# differs significantly from their training distribution, predictions become unreliable. This tutorial covers six
# approaches to OOD detection.
#
# Reconstruction-based methods:
#
# 1. Standard autoencoder (AE): simple reconstruction-based detection using mean squared error
# 1. Variational autoencoder (VAE): a probabilistic approach with a regularized latent space
# 1. AE with GMM: enhanced detection by modeling the latent space with Gaussian Mixture Models
# 1. VAE with GMM: probabilistic encoding combined with GMM density estimation
#
# Distance-based methods:
#
# 1. KNN with cosine distance: measures angular similarity in learned [embeddings](../concepts/Embeddings.md)
# 1. KNN with Euclidean distance: measures straight-line distance in learned embeddings
#
# For this tutorial, you will use MilitaryVehicles, photographs of 24 types of military land vehicle. You will train the
# detectors on every family except the one that the dataset labels Air Defense. A fielded recognizer can meet four kinds
# of input that differ from its training data, and you will score each of them:
#
# - Noisy frames, such as a sensor fault produces
# - Blurred frames, such as a focus or motion fault produces
# - Aerial maritime frames from a different platform, taken from the SeaDrone dataset
# - An unseen family, the one that the dataset labels Air Defense, which the training left out
#
# Each kind is a different change in the data, and a detector that catches one kind can miss another.

# %% [markdown]
# ## Setup
#
# Install the required packages and import the libraries you need.

# %% [markdown]
# ### How to read the detection rates
#
# The threshold of each detector is set so that about 95% of in-distribution samples are accepted. A detection rate is
# the share of a set of images that the detector flags as OOD. On in-distribution images that share is the false alarm
# rate, which the threshold sets near 5%. A detection rate close to 5% on another set of images means that the detector
# does not separate it from the in-distribution data.

# %%
try:
    import google.colab  # noqa: F401

    # %pip install -q dataeval maite-datasets torchvision plotly
except Exception:
    pass

# %%
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
import torch
from maite_datasets.image_classification import MilitaryVehicles
from maite_datasets.object_detection import SeaDrone
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights, resnet18

import dataeval
from dataeval import Embeddings
from dataeval.extractors import TorchExtractor
from dataeval.shift import OODKNeighbors, OODReconstruction
from dataeval.utils.models import AE, VAE, GMMDensityNet
from dataeval.utils.preprocessing import resize, to_canonical_grayscale

# Set random seeds for reproducibility
dataeval.config.set_seed(173, all_generators=True)

# Set the batch size that the reconstruction detectors use in predict
dataeval.config.set_batch_size(64)

# Set default torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataeval.config.set_device(device)
print(f"Using device: {device}")

# Render the plotly figures below with the plotly JS embedded in the notebook output
pio.renderers.default = "notebook"

# %% [markdown]
# ## Prepare the data
#
# Load the train and test splits of MilitaryVehicles with `MilitaryVehicles` from `maite_datasets`. Turn each photograph
# into a 32 by 32 grayscale image with values from 0 to 1, which keeps the models small. The dataset ships with a
# hierarchy that groups its classes into families. Take the four classes of the family labeled Air Defense out of the
# training data and keep their test images as the unseen family.

# %%
input_shape = (1, 32, 32)


def prepare(image):
    """Turn a photograph into a 32 by 32 grayscale image with values from 0 to 1."""
    return (resize(to_canonical_grayscale(np.asarray(image)), 32)[np.newaxis] / 255).astype(np.float32)


def load(image_set):
    """Load a split as prepared images and class labels."""
    dataset = MilitaryVehicles("./data", image_set=image_set, download=True)
    images, labels = [], []
    for image, target, _ in dataset:
        images.append(prepare(image))
        labels.append(int(np.argmax(target)))
    return np.stack(images), np.asarray(labels), dataset


train_images, train_labels, train_data = load("train")
test_images, test_labels, _ = load("test")

held_out_names = train_data.hierarchy["vehicle"]["military"]["land vehicle"]["Air Defense"]
held_out = [train_data.label2index[name] for name in held_out_names]

train_in = train_images[~np.isin(train_labels, held_out)]
test_in = test_images[~np.isin(test_labels, held_out)]
unseen = test_images[np.isin(test_labels, held_out)]

print(f"Held out of training: {held_out_names}")
print(f"Training in-distribution: {train_in.shape}")
print(f"Test in-distribution: {test_in.shape}")
print(f"Test unseen family: {unseen.shape}")

# %% [markdown]
# Build the other three kinds of input. Draw 300 random in-distribution test images, add Gaussian noise to them for the
# noisy frames, and blur them for the blurred frames. The test split is ordered by class, so a random draw keeps the
# images from coming from a few classes. For the aerial frames, load frames from the SeaDrone dataset, which are
# photographs from drones over open water. Load it with `lazy=True` so that it reads only the frames you ask for.

# %%
rng = np.random.default_rng(173)
clean = test_in[np.random.default_rng(1).choice(len(test_in), 300, replace=False)]
noisy = np.clip(clean + rng.normal(0, 0.2, clean.shape), 0, 1).astype(np.float32)
blurred = gaussian_filter(clean, sigma=(0, 0, 2, 2)).astype(np.float32)

seadrone = SeaDrone("./data", image_set="val", download=True, lazy=True)
aerial = np.stack([prepare(seadrone[int(i)][0]) for i in rng.choice(len(seadrone), 300, replace=False)])

scenarios = {"Noisy": noisy, "Blurred": blurred, "Aerial": aerial, "Unseen family": unseen}

# %%
# Show six images of each kind
rows = {"In-distribution": test_in, **scenarios}
fig, axes = plt.subplots(len(rows), 6, figsize=(9, 8))
for row, (title, images) in enumerate(rows.items()):
    for col in range(6):
        axes[row, col].imshow(images[col * 40].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_title(title, loc="left", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## K-nearest neighbors (KNN) for OOD detection
#
# KNN-based OOD detection uses a pretrained model to create learned embeddings. It works by measuring how far test
# samples are from their nearest neighbors in the training data. Samples that are far from all training samples are
# likely OOD.
#
# Use it as a baseline for OOD detection with interpretable distance-based scoring. It needs embeddings from a model.
#
# :::{note}
# KNN performance depends on the quality of the embeddings you provide:
#
# - Better embeddings give better OOD detection, so use task-specific, well-trained models
# - For images, use ResNets, Vision Transformers (ViT), CLIP, or custom CNNs trained on similar data
# - For text, use BERT, sentence transformers, or domain-specific language models
# - For time series, use LSTMs or Transformers trained on temporal data
# - For tabular data, use MLPs or autoencoders trained on your feature space
# :::
#
# This tutorial uses a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
# Torchvision. Point a {class}`.TorchExtractor` at its penultimate `avgpool` layer. The `transforms` repeat the
# grayscale channel three times and apply the standard resize and normalize preprocessing of the model. For other ways
# to build embeddings, see [Create embeddings](./h2_create_embeddings.py).

# %%
weights = ResNet18_Weights.DEFAULT
extractor = TorchExtractor(
    resnet18(weights=weights, progress=False),
    transforms=lambda image: weights.transforms()(image.expand(3, -1, -1)),
    layer_name="avgpool",
)

# Get embeddings for the training data, the in-distribution test data and each scenario
train_emb = Embeddings(train_in, extractor=extractor, batch_size=64)
test_emb = Embeddings(test_in, extractor=extractor, batch_size=64)
scenario_emb = {name: Embeddings(images, extractor=extractor, batch_size=64) for name, images in scenarios.items()}
print(f"Training embeddings shape: {train_emb.shape}")


# %%
def accepted(detector, data):
    """Percent of the data that a fitted detector accepts as in-distribution."""
    return 100 * (1 - detector.predict(data).is_ood.mean())


# Create the KNN detector with cosine distance and fit it on the training embeddings
ood_knn_cos = OODKNeighbors(k=10, distance_metric="cosine", threshold_perc=95.0)
ood_knn_cos.fit(train_emb)
print(f"KNN cosine accepts {accepted(ood_knn_cos, test_emb):.1f}% of the in-distribution test images")

# %% [markdown]
# ### KNN with Euclidean distance
#
# Cosine distance compares the direction of two embeddings and ignores their length. Euclidean distance compares their
# positions, so it also depends on their length. Fit a second detector with Euclidean distance on the same embeddings.

# %%
ood_knn_euc = OODKNeighbors(k=10, distance_metric="euclidean", threshold_perc=95.0)
ood_knn_euc.fit(train_emb)
print(f"KNN Euclidean accepts {accepted(ood_knn_euc, test_emb):.1f}% of the in-distribution test images")

# %% [markdown]
# ## Standard autoencoder (AE) for OOD detection
#
# The simplest approach uses a standard autoencoder that learns to reconstruct normal (in-distribution) images. When
# presented with OOD data, reconstruction error is expected to increase, which signals anomalous samples.
#
# Use it for fast, simple OOD detection when the OOD data is hard to reconstruct, and you do not need probabilistic
# interpretations. Later sections show which inputs it misses.
#
# :::{note}
# This tutorial uses a simple, generic AE architecture provided by DataEval for demonstration purposes. In production:
#
# - Design architectures for your data type: CNNs for images, LSTMs or Transformers for sequences, MLPs for tabular data
# - Match the complexity to your problem: deeper networks for complex data, simpler ones for basic patterns
# - Tune hyperparameters such as the latent dimension size, layer widths and activation functions
# - Your model choice significantly impacts OOD detection performance
# :::
#
# The DataEval `OODReconstruction` class works with any PyTorch model you provide. Customize it for best results.

# %%
# Create and configure the autoencoder
ae_model = AE(input_shape=input_shape)

# Configure training parameters
config = OODReconstruction.Config(
    epochs=3,
    batch_size=64,
    threshold_perc=95.0,  # 95% of training data considered normal
)

# Initialize the OOD detector and train it on the in-distribution data
ood_ae = OODReconstruction(ae_model, device=device, config=config)
print(f"Model type detected: {ood_ae.model_type}")
print(f"Using GMM: {ood_ae.use_gmm}")
ood_ae.fit(train_in)
print(f"AE accepts {accepted(ood_ae, test_in):.1f}% of the in-distribution test images")

# %% [markdown]
# ## Variational autoencoder (VAE) for OOD detection
#
# VAEs learn a probabilistic latent representation, which gives a more structured latent space than a standard AE.
#
# Use it when you need that latent representation or when your in-distribution data has significant variance. In this
# tutorial the plain VAE flags the fewest inputs of the six detectors.

# %%
vae_model = VAE(input_shape=input_shape)
ood_vae = OODReconstruction(vae_model, device=device, config=config)
print(f"Model type detected: {ood_vae.model_type}")
print(f"Using GMM: {ood_vae.use_gmm}")
ood_vae.fit(train_in)
print(f"VAE accepts {accepted(ood_vae, test_in):.1f}% of the in-distribution test images")

# %% [markdown]
# ## Autoencoder with GMM for enhanced OOD detection
#
# A Gaussian Mixture Model (GMM) density network scores each sample by how likely its latent code is under a mixture
# fitted to the in-distribution data. The detector combines that energy with the reconstruction error.
#
# Use it when reconstruction error alone is not enough, for example when the OOD data reconstructs well but sits in an
# unusual region of the latent space.
#
# Add a `GMMDensityNet` to the AE, and set the weight of the GMM energy and how the two scores are combined in the
# configuration.

# %%
ae_model_gmm = AE(input_shape=input_shape)
latent_dim = cast(int, ae_model_gmm.encoder.flatten[1].out_features)
gmm_density_net = GMMDensityNet(latent_dim=latent_dim, n_gmm=3)
ae_model_gmm.gmm_density_net = gmm_density_net

config_gmm = OODReconstruction.Config(
    epochs=3,
    batch_size=64,
    threshold_perc=95.0,  # 95% of training data considered normal
    gmm_weight=0.7,  # For GMM models: balance reconstruction (30%) and GMM energy (70%)
    gmm_score_mode="standardized",  # Use z-score normalization for score fusion
)

ood_ae_gmm = OODReconstruction(ae_model_gmm, device=device, config=config_gmm)
print(f"Model type detected: {ood_ae_gmm.model_type}")
print(f"Using GMM: {ood_ae_gmm.use_gmm}")
print(f"Latent dimension: {latent_dim}")
print(f"Number of GMM components: {gmm_density_net.n_gmm}")
ood_ae_gmm.fit(train_in)
print(f"AE + GMM accepts {accepted(ood_ae_gmm, test_in):.1f}% of the in-distribution test images")

# %% [markdown]
# ## VAE with GMM for OOD detection
#
# This combines the probabilistic encoding of a VAE with the density estimate of a GMM. Use it when you want both.

# %%
vae_model_gmm = VAE(input_shape=input_shape)
vae_model_gmm.gmm_density_net = GMMDensityNet(latent_dim=vae_model_gmm.latent_dim, n_gmm=3)

ood_vae_gmm = OODReconstruction(vae_model_gmm, device=device, config=config_gmm)
ood_vae_gmm.fit(train_in)
print(f"VAE + GMM accepts {accepted(ood_vae_gmm, test_in):.1f}% of the in-distribution test images")

# %% [markdown]
# ## Compare all methods
#
# Score the in-distribution test images and the four kinds of input with each detector. The KNN detectors read
# embeddings, and the reconstruction detectors read images. Each cell of the heatmap is the share of that set of images
# that the detector flags as OOD.

# %%
detectors = {
    "KNN Cosine": ood_knn_cos,
    "KNN Euclidean": ood_knn_euc,
    "AE": ood_ae,
    "VAE": ood_vae,
    "AE+GMM": ood_ae_gmm,
    "VAE+GMM": ood_vae_gmm,
}
inputs = {
    "In-distribution": (test_in, test_emb),
    **{name: (images, scenario_emb[name]) for name, images in scenarios.items()},
}

results = {
    name: {kind: detector.predict(emb if name.startswith("KNN") else images) for kind, (images, emb) in inputs.items()}
    for name, detector in detectors.items()
}
rates = pl.DataFrame({
    "Detector": list(results),
    **{kind: [100 * result[kind].is_ood.mean() for result in results.values()] for kind in inputs},
})

(
    px
    .imshow(
        rates.drop("Detector").to_numpy(),
        x=rates.columns[1:],
        y=rates["Detector"].to_list(),
        zmin=0,
        zmax=100,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    .update_traces(texttemplate="%{z:.0f}%")
    .update_coloraxes(colorbar_title="Flagged (%)")
)

# %% [markdown]
# ### Key observations
#
# The in-distribution column is the false alarm rate, and every detector holds it near the 5% that the threshold sets.
# The other columns show which kind of input each detector catches:
#
# - Noisy frames: the AE flags nearly all of them, and the two GMM variants flag most. KNN with cosine distance flags
#   nearly all of them too. The plain VAE and KNN with Euclidean distance flag almost none.
# - Blurred frames and aerial frames: KNN with cosine distance flags nearly all of them. None of the reconstruction
#   detectors flags a meaningful share, and neither does KNN with Euclidean distance.
# - The unseen family: no detector flags more than a small share. The rates stay close to the false alarm rate, so none
#   of the six separates the family labeled Air Defense from the vehicles that they were trained on.
#
# The cosine and Euclidean detectors read the same embeddings and differ by about 90 points on the aerial frames.
# The next cell shows one difference between the embeddings: the embeddings of the noisy, blurred and aerial frames are
# shorter than those of the in-distribution images. Cosine distance ignores length, and Euclidean distance does not.

# %%
lengths = {"In-distribution": test_emb, **scenario_emb}
for kind, emb in lengths.items():
    print(f"{kind:16} mean embedding length: {np.linalg.norm(np.asarray(emb), axis=1).mean():.1f}")

# %% [markdown]
# ### Interpreting score distributions
#
# A detection rate hides how the scores of a detector compare. Plot the scores that each detector gives the
# in-distribution test images and the aerial frames. For a detector that works, the aerial scores sit to the right of
# the in-distribution scores. KNN with cosine distance shows that. For the reconstruction detectors, the aerial scores
# sit to the left. They do not flag the aerial frames because they score them as more familiar than the vehicle
# photographs.

# %%
scores = pl.concat([
    pl.DataFrame({
        "Detector": [name] * len(result[kind].instance_score),
        "Images": [kind] * len(result[kind].instance_score),
        "Score": result[kind].instance_score,
    })
    for name, result in results.items()
    for kind in ("In-distribution", "Aerial")
])

(
    px
    .histogram(
        scores,
        x="Score",
        color="Images",
        facet_col="Detector",
        facet_col_wrap=3,
        barmode="overlay",
        opacity=0.6,
        nbins=40,
        histnorm="probability",
        height=550,
    )
    .for_each_trace(lambda trace: trace.update(bingroup=trace.xaxis))  # bin each detector on its own scale
    .update_xaxes(matches=None)
    .update_yaxes(matches=None)
    .for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
)

# %% [markdown]
# ## Visualize reconstructions
#
# Examine how the AE reconstructs images of each kind. A reconstruction detector works when the reconstruction error
# is higher for OOD images than for in-distribution images.
#
# Note: KNN does not use reconstruction, so this section covers the autoencoder.


# %%
def reconstruct(detector, images):
    """Get reconstructions from the model of a reconstruction detector."""
    detector.model.to(device)
    detector.model.eval()
    with torch.no_grad():
        output = detector.model(torch.from_numpy(images).float().to(device))
        return (output[0] if isinstance(output, tuple) else output).cpu().numpy()


shown = {"In-distribution": clean, "Noisy": noisy, "Blurred": blurred, "Aerial": aerial}
errors = {}
fig, axes = plt.subplots(len(shown), 2, figsize=(4, 8))
for row, (kind, images) in enumerate(shown.items()):
    reconstructions = reconstruct(ood_ae, images)
    errors[kind] = float(np.mean((images - reconstructions) ** 2))
    for col, image in enumerate([images[0], reconstructions[0]]):
        axes[row, col].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_title(kind, loc="left", fontweight="bold")
    axes[row, 1].set_title(f"mean error {errors[kind]:.4f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Understanding reconstructions
#
# The columns show an original image and its AE reconstruction, and the title of each reconstruction gives the mean
# error over 300 images of that kind:
#
# - The AE reconstructs the vehicle photographs with a small error.
# - It cannot reproduce the noise, so the noisy frames have about three times that error, and the detector flags them.
# - It reconstructs the blurred frames and the aerial frames with about a tenth of that error. Both have little fine
#   detail, so they are easier to reconstruct than the photographs that the AE trained on.
#
# Reconstruction error measures how hard an image is to compress, and it does not measure how unfamiliar an image is.
# That is why a reconstruction detector catches noise, and why it can score a different kind of image as more familiar
# than the training data.

# %% [markdown]
# ## What the results show
#
# These rates come from three epochs of training, about 6,800 training images and generic architectures. With more
# training, more data or architectures built for the data, the rates and their order would change. The reconstruction
# detectors are the most sensitive: the rates of the GMM variants on the noisy frames change by several points between
# seeds and devices. The KNN rates change little, because the embedding model is pretrained.
#
# Match the detector to the change you expect:
#
# - Use a reconstruction detector for faults that add detail, such as sensor noise. It misses faults that remove detail.
# - Use distance on good pretrained embeddings for a change of sensor, platform or image quality, and use cosine
#   distance first.
# - Do not rely on any of the six for a new vehicle of a known kind. Detecting it needs an approach that models the
#   classes, which the embeddings and reconstructions here do not.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# The data
assert train_in.shape == (6846, 1, 32, 32)
assert test_in.shape == (1338, 1, 32, 32)
assert unseen.shape == (283, 1, 32, 32)
assert held_out_names == ["30N6E", "Iskander", "Pantsir-S1", "Rs-24"]
assert len(set(train_labels[~np.isin(train_labels, held_out)])) == 20
assert all(len(images) == 300 for images in (noisy, blurred, aerial))

# The false alarm rate stays near the 5% that the threshold sets
in_dist = {name: result["In-distribution"].is_ood.mean() for name, result in results.items()}
assert all(0.03 < rate < 0.10 for rate in in_dist.values())

# The detection rates the text describes
flagged = {name: {kind: 100 * result[kind].is_ood.mean() for kind in inputs} for name, result in results.items()}
assert flagged["AE"]["Noisy"] > 85
assert flagged["AE+GMM"]["Noisy"] > 70
assert flagged["VAE+GMM"]["Noisy"] > 70
assert flagged["KNN Cosine"]["Noisy"] > 85
assert flagged["VAE"]["Noisy"] < 15
assert flagged["KNN Euclidean"]["Noisy"] < 15
assert flagged["KNN Cosine"]["Blurred"] > 90
assert flagged["KNN Cosine"]["Aerial"] > 90
assert all(flagged[name][kind] < 10 for name in ("AE", "VAE", "AE+GMM", "VAE+GMM") for kind in ("Blurred", "Aerial"))
assert all(flagged["KNN Euclidean"][kind] < 15 for kind in ("Blurred", "Aerial"))
assert all(flagged[name]["Unseen family"] < 20 for name in detectors)
assert flagged["KNN Cosine"]["Aerial"] - flagged["KNN Euclidean"]["Aerial"] > 80

# The OOD embeddings are shorter than the in-distribution embeddings
mean_length = {kind: np.linalg.norm(np.asarray(emb), axis=1).mean() for kind, emb in lengths.items()}
assert all(mean_length["In-distribution"] > 1.2 * mean_length[kind] for kind in ("Noisy", "Blurred", "Aerial"))

# The reconstruction detectors score the aerial frames as more familiar than the in-distribution images
mean_score = {name: {kind: result[kind].instance_score.mean() for kind in inputs} for name, result in results.items()}
assert all(
    mean_score[name]["Aerial"] < mean_score[name]["In-distribution"] for name in ("AE", "VAE", "AE+GMM", "VAE+GMM")
)
assert mean_score["KNN Cosine"]["Aerial"] > mean_score["KNN Cosine"]["In-distribution"]

# The AE reconstruction errors
assert errors["Noisy"] > 2.5 * errors["In-distribution"]
assert all(errors[kind] < 0.15 * errors["In-distribution"] for kind in ("Blurred", "Aerial"))

# %% [markdown]
# ## Conclusion
#
# You fit six out-of-distribution detectors on MilitaryVehicles photographs: {class}`.OODKNeighbors` with cosine and
# with Euclidean distance on pretrained embeddings, and {class}`.OODReconstruction` with an autoencoder, a variational
# autoencoder, and each of those with a Gaussian mixture head. You then scored four kinds of input that the detectors
# never saw. Reconstruction caught noise and nothing that has less detail than the training photographs, cosine
# distance caught every change of sensor or platform, and none of the six caught a new family of vehicles.
#
# The detection rates above belong to this data and to three epochs of training. How each detector behaves on yours
# depends on its embeddings or architecture, so run the comparison on a sample of your own data, with the kinds of
# change you expect.
# [Out-of-distribution detection](../concepts/DistributionShift.md#out-of-distribution-detection) describes the methods,
# and [Create embeddings](./h2_create_embeddings.py) shows how to build the embeddings that the KNN detectors
# need.

# %% [markdown]
# ## Next steps
#
# - [Out-of-distribution detection](../concepts/DistributionShift.md#out-of-distribution-detection) — Learn about the
#   reconstruction-based and distance-based methods.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Detect drift on image embeddings and across chunks of
#   incoming data.
# - [Create embeddings](./h2_create_embeddings.py) — Create image embeddings with PyTorch, ONNX or a model-free
#   extractor.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and data monitoring, run this analysis using your own reference and operational
# datasets.
#
# Experiment with these changes:
#
# - Try better embeddings for KNN: ResNet, ViT, CLIP, or domain-specific pretrained models
# - Compare cosine and Euclidean distance on your embeddings to find the best fit
# - Train the models for more epochs and compare the rates
# - Try different `gmm_weight` and `n_gmm` values
# - Design custom architectures for your specific data type, and not generic examples
# - Test the kinds of change you expect on your own data, such as a new sensor, a degraded sensor or a new class
# - Tune `threshold_perc` for your false positive tolerance
