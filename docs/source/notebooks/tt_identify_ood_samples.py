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
# - Train different reconstruction models (AE, VAE) for OOD detection
# - Use Gaussian Mixture Models (GMM) to enhance OOD detection
# - Compare KNN-based OOD detection with cosine and Euclidean distance metrics
# - Compare model performance on different OOD scenarios
# - Visualize reconstruction quality and OOD scores
#
# ## What you will learn
#
# - How to fit the six detectors on in-distribution data and score new data with each
# - How the detection rate changes as the OOD data gets closer to the training data
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
# 1. VAE with GMM: probabilistic encoding combined with GMM for reliable detection
#
# Distance-based methods:
#
# 1. KNN with cosine distance: measures angular similarity in learned [embeddings](../concepts/Embeddings.md)
# 1. KNN with Euclidean distance: measures straight-line distance in learned embeddings
#
# For this tutorial, you will use the MNIST dataset of handwritten digits. You will train models to recognize digits 0-7
# and test their ability to detect digits 8-9 as out-of-distribution samples.

# %% [markdown]
# ## Setup
#
# Install the required packages and import the libraries you need.

# %% [markdown]
# ### Important note on expected results
#
# OOD detection performance depends on how different the OOD data is from the in-distribution data. This tutorial trains
# on digits 0-7 and treats digits 8-9 as out of distribution. That is a hard case: digit 8 shares circular shapes with 0
# and 6, and digit 9 shares curves with 3 and 5. The threshold of each detector is set so that about 95% of
# in-distribution samples are accepted. The "Comparing use cases" section adds an easy and a medium case.

# %%
try:
    import google.colab  # noqa: F401

    # %pip install -q dataeval torchvision
except Exception:
    pass

# %%
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from maite_datasets.image_classification import CIFAR10, MNIST

import dataeval
from dataeval import Embeddings
from dataeval.data import Filter, Limit, Shuffle, View
from dataeval.extractors import TorchExtractor
from dataeval.shift import OODKNeighbors, OODReconstruction
from dataeval.utils.models import AE, VAE, GMMDensityNet
from dataeval.utils.preprocessing import rescale, resize, to_canonical_grayscale

# Set random seeds for reproducibility
dataeval.config.set_seed(173, all_generators=True)

# Set the batch size that the reconstruction detectors use in predict
dataeval.config.set_batch_size(64)

# Set default torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataeval.config.set_device(device)
print(f"Using device: {device}")


# %% [markdown]
# ## Prepare the data
#
# You will load the MNIST dataset and split it into in-distribution (digits 0-7) and out-of-distribution (digits 8-9)
# samples.


# %%
def normalize(x):
    return x.astype(np.float32) / 255.0


in_dist_digits = [0, 1, 2, 3, 4, 5, 6, 7]
out_of_dist_digits = [8, 9]


mnist_train = View(
    MNIST("./data", image_set="train", download=True, transforms=normalize),
    operations=[Filter(classes=in_dist_digits), Shuffle(), Limit(10000)],
)
mnist_test_in = View(
    MNIST("./data", image_set="test", download=True, transforms=normalize),
    operations=[Filter(classes=in_dist_digits), Shuffle(), Limit(1000)],
)
mnist_test_ood = View(
    MNIST("./data", image_set="test", download=True, transforms=normalize),
    operations=[Filter(classes=out_of_dist_digits), Shuffle(), Limit(1000)],
)

print(f"Training set size: {len(mnist_train)}")
print(f"Test set size: {len(mnist_test_in)}")
print(f"Test set size: {len(mnist_test_ood)}")

# Set the input shape (MNIST images are 28x28 grayscale)
input_shape = (1, 28, 28)


# %%
# Extract data and labels from prefiltered datasets
def extract_data_labels(dataset):
    """Extract images and labels from a dataset."""
    data, labels = [], []

    for img, label_probs, _ in dataset:
        label = np.argmax(label_probs)
        data.append(img)
        labels.append(label)

    return np.stack(data), np.asarray(labels)


# Extract training and test data (already filtered for correct classes)
train_in, train_in_labels = extract_data_labels(mnist_train)
test_in, test_in_labels = extract_data_labels(mnist_test_in)
test_ood, test_ood_labels = extract_data_labels(mnist_test_ood)

print(f"Training in-distribution: {train_in.shape}")
print(f"Test in-distribution: {test_in.shape}")
print(f"Test out-of-distribution: {test_ood.shape}")

# %%
# Visualize some in-distribution and OOD samples
fig, axes = plt.subplots(2, 8, figsize=(12, 3))

# Show in-distribution samples (0-7) - one of each digit
for digit in range(8):
    # Find the first occurrence of this digit
    idx = (train_in_labels == digit).nonzero()[0][0]
    axes[0, digit].imshow(train_in[idx].squeeze(), cmap="gray")
    axes[0, digit].axis("off")
    axes[0, digit].set_title(f"Digit {digit}")

# Show OOD samples (8-9) - 4 of each
for i in range(8):
    digit = 8 if i < 4 else 9
    idx = (test_ood_labels == digit).nonzero()[0][(i % 4) * 50]
    axes[1, i].imshow(test_ood[idx].squeeze(), cmap="gray")
    axes[1, i].axis("off")
    if i % 4 == 0:
        axes[1, i].set_title(f"Digit {digit} (OOD)", color="red")

axes_text_kwargs = {"ha": "right", "va": "center", "fontsize": 12, "fontweight": "bold"}
axes[0, 0].text(-0.5, 0.5, "In-Dist\n(Train)", transform=axes[0, 0].transAxes, **axes_text_kwargs)
axes[1, 0].text(-0.5, 0.5, "OOD\n(Test)", transform=axes[1, 0].transAxes, **axes_text_kwargs)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## K-nearest neighbors (KNN) for OOD detection
#
# KNN-based OOD detection uses a pretrained model to create learned embeddings. It works by measuring how far test
# samples are from their nearest neighbors in the training data. Samples that are far from all training samples are
# likely OOD.
#
# Use it as a fast baseline for OOD detection without model training, with interpretable distance-based scoring.
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
# This tutorial trains a simple CNN for demonstration, but using pretrained models (for example, an ImageNet-pretrained
# ResNet) would likely improve the results.


# %%
# Define a simple CNN for learning embeddings
class EmbeddingNet(torch.nn.Module):
    """Simple CNN that learns embeddings for digit classification."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Convolutional layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 28x28 -> 14x14
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        # Embedding layer
        self.embedding = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_dim),
        )

        # Classification head (for training only)
        self.classifier = torch.nn.Linear(embedding_dim, 8)  # 8 digit classes (0-7)

    def forward(self, x, return_embedding=False):
        """Forward pass. Returns embeddings if return_embedding=True, else logits."""
        emb = self.embedding(self.conv_layers(x))
        return emb if return_embedding else self.classifier(emb)


# %%
# Create and train the embedding model
embedding_model = EmbeddingNet(embedding_dim=64).to(device)
optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("Training embedding model for digit classification...")
print(f"Embedding dimension: {embedding_model.embedding_dim}")

# Train for a few epochs
num_epochs = 3
batch_size = 256

for epoch in range(num_epochs):
    embedding_model.train()
    total_loss, correct, total = 0, 0, 0

    # Create batches
    num_batches = len(train_in) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_imgs = torch.as_tensor(train_in[start_idx:end_idx], device=device)
        batch_labels = torch.as_tensor(train_in_labels[start_idx:end_idx], device=device)

        # Forward pass
        optimizer.zero_grad()
        logits = embedding_model(batch_imgs)
        loss = criterion(logits, batch_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("✓ Embedding model trained!")

# %%
# Create extractor using the trained embedding model
knn_extractor = TorchExtractor(embedding_model)

# Get embeddings for all datasets
print("Extracting embeddings...")
train_in_emb = Embeddings(train_in, extractor=knn_extractor, batch_size=64)
test_in_emb = Embeddings(test_in, extractor=knn_extractor, batch_size=64)
test_ood_emb = Embeddings(test_ood, extractor=knn_extractor, batch_size=64)

print(f"Training embeddings shape: {train_in_emb.shape}")
print(f"Test in-dist embeddings shape: {test_in_emb.shape}")
print(f"Test OOD embeddings shape: {test_ood_emb.shape}")

# %%
# Create KNN detector with learned embeddings
ood_knn_cos = OODKNeighbors(k=10, distance_metric="cosine", threshold_perc=95.0)

print("\nFitting KNN detector with learned embeddings...")
ood_knn_cos.fit(train_in_emb)
print("Done!")

# %%
# Get predictions with learned embeddings
knn_cos_result_in = ood_knn_cos.predict(test_in_emb)
knn_cos_result_ood = ood_knn_cos.predict(test_ood_emb)

# Calculate detection accuracy
in_acc_knn = 100 * (1 - knn_cos_result_in.is_ood.mean())
ood_rate_knn = 100 * knn_cos_result_ood.is_ood.mean()

print("\n--- KNN Cosine (Embeddings) Results ---")
print(f"In-distribution correctly identified: {in_acc_knn:.1f}%")
print(f"OOD samples detected: {ood_rate_knn:.1f}%")
print(f"Average score (in-dist): {knn_cos_result_in.instance_score.mean():.4f}")
print(f"Average score (OOD): {knn_cos_result_ood.instance_score.mean():.4f}")

# %% [markdown]
# ### KNN with Euclidean distance
#
# Euclidean distance measures the straight-line distance between points in the embedding space. Unlike cosine
# similarity, which only considers the angle between vectors, Euclidean distance also accounts for their magnitude. This
# makes it better suited for embeddings where the scale of the vectors carries meaningful information.

# %%
# Create KNN detector with Euclidean distance
ood_knn_euc = OODKNeighbors(k=10, distance_metric="euclidean", threshold_perc=95.0)

print("Fitting KNN (Euclidean) detector with learned embeddings...")
ood_knn_euc.fit(train_in_emb)
print("Done!")

# %%
# Get predictions with Euclidean distance
knn_euc_result_in = ood_knn_euc.predict(test_in_emb)
knn_euc_result_ood = ood_knn_euc.predict(test_ood_emb)

# Calculate detection accuracy
in_acc_knn_euc = 100 * (1 - knn_euc_result_in.is_ood.mean())
ood_rate_knn_euc = 100 * knn_euc_result_ood.is_ood.mean()

print("\n--- KNN Euclidean (Embeddings) Results ---")
print(f"In-distribution correctly identified: {in_acc_knn_euc:.1f}%")
print(f"OOD samples detected: {ood_rate_knn_euc:.1f}%")
print(f"Average score (in-dist): {knn_euc_result_in.instance_score.mean():.4f}")
print(f"Average score (OOD): {knn_euc_result_ood.instance_score.mean():.4f}")

# %% [markdown]
# ## Standard autoencoder (AE) for OOD detection
#
# The simplest approach uses a standard autoencoder that learns to reconstruct normal (in-distribution) images. When
# presented with OOD data, reconstruction error increases, which signals anomalous samples.
#
# Use it for fast, simple OOD detection when your distributions are well separated and you do not need probabilistic
# interpretations.
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

# Initialize OOD detector
ood_ae = OODReconstruction(ae_model, device=device, config=config)

print("Training Standard Autoencoder...")
print(f"Model type detected: {ood_ae.model_type}")
print(f"Using GMM: {ood_ae.use_gmm}")

# %%
# Train the model on in-distribution data
ood_ae.fit(train_in)

# %%
# Get predictions
ae_result_in = ood_ae.predict(test_in)
ae_result_ood = ood_ae.predict(test_ood)

# Calculate detection accuracy
in_acc_ae = 100 * (1 - ae_result_in.is_ood.mean())
ood_rate_ae = 100 * ae_result_ood.is_ood.mean()

print("\n--- Standard AE Results ---")
print(f"In-distribution correctly identified: {in_acc_ae:.1f}%")
print(f"OOD samples detected: {ood_rate_ae:.1f}%")
print(f"Average score (in-dist): {ae_result_in.instance_score.mean():.4f}")
print(f"Average score (OOD): {ae_result_ood.instance_score.mean():.4f}")

# Validation: Check if OOD scores are higher than in-dist scores
score_separation = ae_result_ood.instance_score.mean() - ae_result_in.instance_score.mean()
print(f"\nScore separation (OOD - In-Dist): {score_separation:.4f}")
if score_separation > 0:
    print("✓ Expected: OOD samples have higher scores than in-distribution samples")
else:
    print("⚠ Warning: OOD scores should be higher than in-distribution scores")

# Check if we're near the target threshold
if 90 <= in_acc_ae <= 98:
    print(f"✓ Expected: ~95% of in-distribution samples correctly identified (got {in_acc_ae:.1f}%)")
else:
    print(f"⚠ Note: Expected ~95% in-dist accuracy, got {in_acc_ae:.1f}%")

# %% [markdown]
# ## Variational autoencoder (VAE) for OOD detection
#
# VAEs learn a probabilistic latent representation, which provides better generalization and more structured latent
# spaces compared to standard AEs. This can improve OOD detection, especially when in-distribution data has high
# variability.
#
# Use it when you need a more robust latent representation or when your in-distribution data has significant variance.

# %%
# Create and configure the VAE
vae_model = VAE(input_shape=input_shape)

# Initialize OOD detector (auto-detects as VAE)
ood_vae = OODReconstruction(vae_model, device=device, config=config)

print("Training Variational Autoencoder...")
print(f"Model type detected: {ood_vae.model_type}")
print(f"Using GMM: {ood_vae.use_gmm}")

# %%
# Train the VAE
ood_vae.fit(train_in)

# %%
# Evaluate VAE performance
vae_result_in = ood_vae.predict(test_in)
vae_result_ood = ood_vae.predict(test_ood)

in_acc_vae = 100 * (1 - vae_result_in.is_ood.mean())
ood_rate_vae = 100 * vae_result_ood.is_ood.mean()

print("\n--- VAE Results ---")
print(f"In-distribution correctly identified: {in_acc_vae:.1f}%")
print(f"OOD samples detected: {ood_rate_vae:.1f}%")
print(f"Average score (in-dist): {vae_result_in.instance_score.mean():.4f}")
print(f"Average score (OOD): {vae_result_ood.instance_score.mean():.4f}")

# %% [markdown]
# ## Autoencoder with GMM for enhanced OOD detection
#
# Adding a Gaussian Mixture Model (GMM) to the latent space provides an additional signal for OOD detection. The GMM
# models the density of the latent representations, and samples with low density are likely to be OOD. This combines
# reconstruction error with density estimation using **sensor fusion**: both components are standardized (z-score
# normalized) and combined with configurable weighting.
#
# Use it when you need higher detection accuracy and have complex in-distribution data that naturally clusters into
# multiple groups.
#
# :::{note}
# The GMM fusion parameters (`gmm_weight` and `gmm_score_mode`) significantly impact performance. The default
# `gmm_weight=0.7` favors the GMM component, which typically works well. Experiment with values in [0.5, 0.9] for your
# data.
# :::

# %%
# Create AE with GMM density network
# The latent dimension is auto-computed by AE
ae_model_gmm = AE(input_shape=input_shape)
latent_dim = cast(int, ae_model_gmm.encoder.flatten[1].out_features)

# Create GMM density network with 3 components
gmm_density_net = GMMDensityNet(latent_dim=latent_dim, n_gmm=3)
ae_model_gmm.gmm_density_net = gmm_density_net

# Configure training parameters
config_gmm = OODReconstruction.Config(
    epochs=3,
    batch_size=64,
    threshold_perc=95.0,  # 95% of training data considered normal
    gmm_weight=0.7,  # For GMM models: balance reconstruction (30%) and GMM energy (70%)
    gmm_score_mode="standardized",  # Use z-score normalization for score fusion
)

# Initialize OOD detector (auto-detects GMM usage)
ood_ae_gmm = OODReconstruction(ae_model_gmm, device=device, config=config_gmm)

print("Training Autoencoder with GMM...")
print(f"Model type detected: {ood_ae_gmm.model_type}")
print(f"Using GMM: {ood_ae_gmm.use_gmm}")
print(f"Latent dimension: {latent_dim}")
print(f"Number of GMM components: {gmm_density_net.n_gmm}")

# %%
# Train the AE+GMM model
ood_ae_gmm.fit(train_in)

# %%
# Evaluate AE+GMM performance
ae_gmm_result_in = ood_ae_gmm.predict(test_in)
ae_gmm_result_ood = ood_ae_gmm.predict(test_ood)

in_acc_ae_gmm = 100 * (1 - ae_gmm_result_in.is_ood.mean())
ood_rate_ae_gmm = 100 * ae_gmm_result_ood.is_ood.mean()

print("\n--- AE + GMM Results ---")
print(f"In-distribution correctly identified: {in_acc_ae_gmm:.1f}%")
print(f"OOD samples detected: {ood_rate_ae_gmm:.1f}%")
print(f"Average score (in-dist): {ae_gmm_result_in.instance_score.mean():.4f}")
print(f"Average score (OOD): {ae_gmm_result_ood.instance_score.mean():.4f}")

# %% [markdown]
# ## VAE with GMM for maximum robustness
#
# Combining VAE's probabilistic latent space with GMM density estimation provides the most sophisticated OOD detection
# approach. This is particularly effective when you need high reliability and have sufficient computational resources.
#
# Use it in production systems where false negatives (missing OOD samples) are costly, and you need maximum detection
# reliability.

# %%
# Create VAE with GMM density network
vae_model_gmm = VAE(input_shape=input_shape)
vae_latent_dim = vae_model_gmm.latent_dim

# Create GMM density network
gmm_density_net_vae = GMMDensityNet(latent_dim=vae_latent_dim, n_gmm=3)
vae_model_gmm.gmm_density_net = gmm_density_net_vae

# Initialize OOD detector
ood_vae_gmm = OODReconstruction(vae_model_gmm, device=device, config=config_gmm)

print("Training VAE with GMM...")
print(f"Model type detected: {ood_vae_gmm.model_type}")
print(f"Using GMM: {ood_vae_gmm.use_gmm}")
print(f"Latent dimension: {vae_latent_dim}")
print(f"Number of GMM components: {gmm_density_net_vae.n_gmm}")

# %%
# Train the VAE+GMM model
ood_vae_gmm.fit(train_in)

# %%
# Evaluate VAE+GMM performance
vae_gmm_result_in = ood_vae_gmm.predict(test_in)
vae_gmm_result_ood = ood_vae_gmm.predict(test_ood)

in_acc_vae_gmm = 100 * (1 - vae_gmm_result_in.is_ood.mean())
ood_rate_vae_gmm = 100 * vae_gmm_result_ood.is_ood.mean()

print("\n--- VAE + GMM Results ---")
print(f"In-distribution correctly identified: {in_acc_vae_gmm:.1f}%")
print(f"OOD samples detected: {ood_rate_vae_gmm:.1f}%")
print(f"Average score (in-dist): {vae_gmm_result_in.instance_score.mean():.4f}")
print(f"Average score (OOD): {vae_gmm_result_ood.instance_score.mean():.4f}")

# %% [markdown]
# ## Compare all methods
#
# Now visualize and compare the performance of all six approaches.

# %%
# Summary comparison
methods = ["KNN\nCosine", "KNN\nEuclidean", "AE", "VAE", "AE+GMM", "VAE+GMM"]
in_dist_acc = [in_acc_knn, in_acc_knn_euc, in_acc_ae, in_acc_vae, in_acc_ae_gmm, in_acc_vae_gmm]
ood_detect = [ood_rate_knn, ood_rate_knn_euc, ood_rate_ae, ood_rate_vae, ood_rate_ae_gmm, ood_rate_vae_gmm]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Plot in-distribution accuracy
colors = ["#3498db", "#2980b9", "#9b59b6", "#8e44ad", "#2ecc71", "#e74c3c"]
bars1 = ax1.bar(methods, in_dist_acc, color=colors)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("In-Distribution Samples Correctly Identified", fontsize=14, fontweight="bold")
ax1.set_ylim([0, 105])
ax1.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="Target: 95%")
ax1.legend()
ax1.tick_params(axis="x", rotation=0)
text_kwargs = {"ha": "center", "va": "bottom", "fontsize": 9, "fontweight": "bold"}
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%", **text_kwargs)

# Plot OOD detection rate
bars2 = ax2.bar(methods, ood_detect, color=colors)
ax2.set_ylabel("Detection Rate (%)", fontsize=12)
ax2.set_title("Out-of-Distribution Samples Detected", fontsize=14, fontweight="bold")
ax2.set_ylim([0, 105])
ax2.tick_params(axis="x", rotation=0)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%", **text_kwargs)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Key observations
#
# The thresholds accept about 95% of in-distribution samples. The cell prints the range of the in-distribution rates and
# the OOD detection rates, highest first.

# %%
print(f"In-distribution samples accepted: {min(in_dist_acc):.1f}% to {max(in_dist_acc):.1f}% (target 95%)")
print("OOD samples detected, highest first:")
for name, rate in sorted(zip(methods, ood_detect, strict=True), key=lambda pair: pair[1], reverse=True):
    print(f"  {name.replace(chr(10), ' '):15} {rate:.1f}%")

# %% [markdown]
# Digits 8 and 9 share features with 0-7, so no detector flags more than half of them. The two distance metrics for KNN
# differ by more than ten points here, and the plain VAE detects the fewest.

# %%
# Visualize OOD score distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
results = [
    (knn_cos_result_in, knn_cos_result_ood, "KNN (Cosine)"),
    (ae_result_in, ae_result_ood, "AE"),
    (vae_result_in, vae_result_ood, "VAE"),
    (knn_euc_result_in, knn_euc_result_ood, "KNN (Euclidean)"),
    (ae_gmm_result_in, ae_gmm_result_ood, "AE + GMM"),
    (vae_gmm_result_in, vae_gmm_result_ood, "VAE + GMM"),
]

for idx, result in enumerate(results):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    result_in, result_ood, title = result

    # Plot histograms
    ax.hist(result_in.instance_score, bins=50, alpha=0.6, label="In-Distribution", color="blue")
    ax.hist(result_ood.instance_score, bins=50, alpha=0.6, label="Out-of-Distribution", color="red")

    ax.set_xlabel("OOD Score", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Interpreting score distributions
#
# Look at how far apart the blue (in-distribution) and red (OOD) histograms sit. Clear separation means the detector
# tells the two apart, and overlap means it cannot. The printed average scores show that every detector scores the OOD
# samples higher than the in-distribution samples on average. The overlap is large here because digits 8 and 9 share
# features with the training digits.

# %% [markdown]
# ## Visualize reconstructions
#
# Examine how reconstruction-based models reconstruct in-distribution and out-of-distribution samples. Good OOD
# detection should show clear degradation in reconstruction quality for OOD samples.
#
# Note: KNN does not use reconstruction, so this section focuses on the autoencoder-based methods.


# %%
# Helper function to get reconstructions
def get_reconstructions(model, data, device):
    """Get reconstructions from a model."""
    model.model.to(device)
    model.model.eval()
    with torch.no_grad():
        data_tensor = torch.from_numpy(data).float().to(device)
        output = model.model(data_tensor)
        reconstruction = output[0] if isinstance(output, tuple) else output
        return reconstruction.cpu().numpy()


# Get samples: 1 in-dist, 1 OOD stacked as rows
n_samples = 2
originals = np.concatenate([test_in[:n_samples], test_ood[:n_samples]], axis=0)  # (4, 1, 28, 28)

# Get reconstructions for all samples
recon_ae = get_reconstructions(ood_ae, originals, device)  # (4, 1, 28, 28)
recon_vae_gmm = get_reconstructions(ood_vae_gmm, originals, device)  # (4, 1, 28, 28)

# Stack columns: Original, AE, VAE -> shape (4, 3, 1, 28, 28)
recon_grid = np.stack([originals, recon_ae, recon_vae_gmm], axis=1)

# %%
# Visualize reconstructions: rows = samples, columns = Original/AE/VAE
fig, axes = plt.subplots(4, 3, figsize=(6, 8))

# Column titles
col_titles = ["Original", "AE", "VAE+GMM"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=12, fontweight="bold")

# Row labels
row_labels = ["In-Dist", "In-Dist", "OOD", "OOD"]

# Plot each cell using recon_grid[row, col]
for i, label in enumerate(row_labels):
    # Add row label
    color = "darkgreen" if "In-Dist" in label else "darkred"
    axes[i, 0].text(
        -0.3,
        0.5,
        label,
        transform=axes[i, 0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=color,
    )

    for j in range(3):
        axes[i, j].imshow(recon_grid[i, j].squeeze(), cmap="gray")
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Understanding reconstructions
#
# What to observe:
#
# - The columns show the original image, the AE reconstruction and the VAE+GMM reconstruction.
# - Rows 1-2 are in-distribution samples (digits 5 and 4).
# - Rows 3-4 are out-of-distribution samples (digit 8).
#
# Expected reconstruction behavior:
#
# - In-distribution samples: the model has learned these patterns, so you expect good reconstruction and low error.
# - OOD samples: the model has not seen these patterns, so you expect worse reconstruction and high error.
#
# Note: The degree of degradation depends on the similarity between the in-distribution and OOD data:
#
# - Digits 8 and 9 share some features with 0-7 (curves, circles).
# - So reconstructions may still look reasonable but will have higher error.
# - More distinct OOD data (for example, letters instead of digits) would show clearer degradation.

# %% [markdown]
# ## Comparing use cases - when does each method excel?
#
# :::{note}
# The results reflect limited training and generic models.
# :::
#
# This comparison uses:
#
# - Only 3 epochs for AE/VAE training and KNN embedding model training (production typically needs 10-50+ epochs)
# - A small sample size: 10K training, 3K test samples
# - Generic model architectures: simple CNNs not optimized for MNIST
# - A fast demonstration, prioritized over optimal performance
#
# This means:
#
# - The results show what happens with minimal training and generic models, which is useful for quick prototypes
# - Model architecture matters: custom architectures designed for your data type (images, time series, tabular) will
#   perform significantly better
# - With proper training and tuning and domain-specific architectures, the performance rankings may change significantly
# - Use these results as a starting point, not definitive guidance
#
# :::{note}
# The AE, VAE, and GMM methods use models you provide. Performance heavily depends on:
#
# - Choosing appropriate architectures for your data type and complexity
# - Proper hyperparameter tuning (latent dimensions, layer sizes, activation functions)
# - Sufficient training epochs and data
# - Appropriate loss functions and regularization
# :::
#
# The simple models used here serve as examples. Real applications should use architectures targeted to the specific
# scenario.
#
# Test each method on different OOD scenarios to see its strengths and weaknesses in this limited-training setting.
#
# You create three OOD scenarios with increasing difficulty:
#
# 1. Easy OOD: CIFAR10 natural images (converted to grayscale 28x28), which are completely different from digits
# 1. Medium OOD: rotated digits, which are the same objects in a different orientation
# 1. Hard OOD: digits 8-9 have features similar to the training data (the current scenario)

# %%
# Create different OOD scenarios

# Scenario 1: Easy OOD - CIFAR10 (completely different domain: natural images vs digits)
# Load CIFAR10 and convert to match MNIST format
cifar_dataset = CIFAR10("./data", image_set="test", download=True)
easy_ood_list = []
for i in range(500):
    img = cifar_dataset[i][0]
    img_gray = resize(to_canonical_grayscale(rescale(img, 8)), 28)[np.newaxis, :]
    easy_ood_list.append(normalize(img_gray))
easy_ood = np.stack(easy_ood_list)

# Scenario 2: Medium OOD - Rotated digits (same domain, different transformation)
medium_ood = np.rot90(test_in[:500], k=1, axes=(2, 3)).copy()

# Scenario 3: Hard OOD - Digits 8-9 (already created as test_ood_subset)
hard_ood = test_ood

# Get embeddings for all OOD scenarios (reuse the same extractor)
easy_ood_emb = Embeddings(easy_ood, extractor=knn_extractor, batch_size=64)
medium_ood_emb = Embeddings(medium_ood, extractor=knn_extractor, batch_size=64)
hard_ood_emb = Embeddings(hard_ood, extractor=knn_extractor, batch_size=64)

print("Created three OOD scenarios:")
print(f"1. Easy (CIFAR10 → grayscale): {easy_ood.shape}")
print(f"2. Medium (Rotated digits): {medium_ood.shape}")
print(f"3. Hard (Digits 8-9): {hard_ood.shape}")

# %%
# Visualize the different OOD scenarios
fig, axes = plt.subplots(3, 5, figsize=(12, 7))

ood_by_scenario = [easy_ood, medium_ood, hard_ood]
ood_title = [("Easy OOD (CIFAR10)", "red"), ("Medium OOD (Rotated)", "orange"), ("Hard OOD (Digits 8-9)", "darkred")]

# Easy OOD - CIFAR10 (grayscale)
for i in range(5):
    for j in range(3):
        if i == 0:
            axes[j, 0].set_title(ood_title[j][0], fontweight="bold", color=ood_title[j][1])
        axes[j, i].imshow(ood_by_scenario[j][i * 20].squeeze(), cmap="gray")
        axes[j, i].axis("off")

plt.tight_layout()
plt.show()

# %%
# Evaluate all models on all three OOD scenarios
models = {
    "KNN Cosine": ood_knn_cos,
    "KNN Euclidean": ood_knn_euc,
    "AE": ood_ae,
    "VAE": ood_vae,
    "AE+GMM": ood_ae_gmm,
    "VAE+GMM": ood_vae_gmm,
}

scenarios = {
    "Easy (CIFAR10)": (easy_ood, easy_ood_emb),
    "Medium (Rotated)": (medium_ood, medium_ood_emb),
    "Hard (Digits 8-9)": (hard_ood, hard_ood_emb),
}

# Store results
results_matrix = {}

for model_name, model in models.items():
    results_matrix[model_name] = {}
    for scenario_name, (ood_data, ood_data_emb) in scenarios.items():
        # Use appropriate data format
        data_to_use = ood_data_emb if model_name.startswith("KNN") else ood_data

        result = model.predict(data_to_use)
        detection_rate = 100 * result.is_ood.mean()
        results_matrix[model_name][scenario_name] = detection_rate

# %%
# Create heatmap visualization
fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(results_matrix.keys())
scenario_names = list(scenarios.keys())

# Create matrix for heatmap
data = np.array([[results_matrix[model][scenario] for scenario in scenario_names] for model in model_names])

im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=100)

# Set ticks and labels
ax.set_xticks(np.arange(len(scenario_names)))
ax.set_yticks(np.arange(len(model_names)))
ax.set_xticklabels(scenario_names)
ax.set_yticklabels(model_names)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(model_names)):
    for j in range(len(scenario_names)):
        text = ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center", color="black", fontweight="bold")

ax.set_title("OOD Detection Rate by Model and Scenario", fontsize=14, fontweight="bold", pad=20)
fig.colorbar(im, ax=ax, label="Detection Rate (%)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### What the results show
#
# The table lists the share of each OOD scenario that each detector flags.

# %%
print(f"{'':15}" + "".join(f"{name:>20}" for name in scenario_names))
for model_name, rates in results_matrix.items():
    print(f"{model_name:15}" + "".join(f"{rates[name]:>19.1f}%" for name in scenario_names))

# %% [markdown]
# - Detection falls as the OOD data gets closer to the training data: every detector flags more of the CIFAR10 images
#   than of the digits 8-9, and every detector flags more than 80% of the CIFAR10 images.
# - On the rotated digits, the two detectors with a GMM head flag the most and the plain VAE flags the fewest.
# - On digits 8-9, KNN with cosine distance flags the most and the plain VAE flags the fewest.
#
# These rates come from three epochs of training, 10,000 training samples and generic architectures. With more training,
# more data or architectures built for the data, the rates and their order would change.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert all(abs(rate - 95) < 2.5 for rate in in_dist_acc)
assert max(ood_detect) < 50
assert abs(ood_detect[0] - ood_detect[1]) > 10
assert ood_detect[3] == min(ood_detect)
assert all(r_in.instance_score.mean() < r_ood.instance_score.mean() for r_in, r_ood, _ in results)

easy, medium, hard = ({model: rates[scenario] for model, rates in results_matrix.items()} for scenario in scenarios)
assert all(easy[model] > hard[model] for model in results_matrix)
assert all(rate > 80 for rate in easy.values())
assert set(sorted(medium, key=lambda model: medium[model], reverse=True)[:2]) == {"AE+GMM", "VAE+GMM"}
assert min(medium, key=lambda model: medium[model]) == "VAE"
assert max(hard, key=lambda model: hard[model]) == "KNN Cosine"
assert min(hard, key=lambda model: hard[model]) == "VAE"

# %% [markdown]
# ## Conclusion
#
# You fit six out-of-distribution detectors on MNIST digits 0-7: {class}`.OODKNeighbors` with cosine and with Euclidean
# distance, and {class}`.OODReconstruction` with an autoencoder, a variational autoencoder, and each of those with a
# Gaussian mixture head. You then scored digits 8-9 and two easier scenarios with each.
#
# The detection rates above belong to this data and to three epochs of training. How each detector behaves on yours
# depends on its embeddings or architecture, so run the comparison on a sample of your own data.
# [Out-of-distribution detection](../concepts/DistributionShift.md#out-of-distribution-detection) describes the methods,
# and [How to create embeddings](./h2_create_embeddings.py) shows how to build the embeddings that the KNN detectors
# need.

# %% [markdown]
# ## Next steps
#
# - [Out-of-distribution detection](../concepts/DistributionShift.md#out-of-distribution-detection) — Learn about the reconstruction-based and distance-based methods.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Detect drift on image embeddings and across chunks of incoming data.
# - [How to create embeddings](./h2_create_embeddings.py) — Create image embeddings with PyTorch, ONNX or a model-free extractor.

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
# - Test different OOD scenarios on your own data, with varying difficulty levels
# - Tune `threshold_perc` for your false positive tolerance
# - Use pretrained models instead of training from scratch
