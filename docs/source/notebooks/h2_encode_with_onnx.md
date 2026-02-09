---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: dataeval
  language: python
  name: python3
---

# How to encode images with ONNX models

+++

## _Problem Statement_

When working with image datasets, generating embeddings is a common first step for many analysis tasks like clustering,
duplicate detection, and coverage analysis. While PyTorch models are widely used, ONNX (Open Neural Network Exchange)
provides a framework-agnostic format that offers portability and often better inference performance.

DataEval's `OnnxExtractor` allows you to use any ONNX model to generate embeddings from your image datasets.

+++

### _When to use_

Use the `OnnxExtractor` when you want to:

- Generate embeddings using a pre-trained ONNX model
- Work with models exported from various frameworks (PyTorch, TensorFlow, etc.)
- Leverage optimized inference without framework dependencies

+++

### _What you will need_

1. An image dataset (we'll use VOC2012)
2. An ONNX model that outputs embeddings
3. A Python environment with the following packages installed:
   - `dataeval`
   - `onnxruntime` (or `onnxruntime-gpu` for GPU support)
   - `onnx` (for model preparation utilities)
   - `maite-datasets`

+++

## _Getting Started_

Let's import the required libraries needed to set up a minimal working example.

```{code-cell} ipython3
try:
    import google.colab  # noqa: F401

    %pip install -q dataeval[onnx] maite-datasets opencv-python-headless
except Exception:
    pass
```

```{code-cell} ipython3
import os

import cv2
import numpy as np
import requests
from maite_datasets.object_detection import VOCDetection

from dataeval import Embeddings
from dataeval.extractors import OnnxExtractor
from dataeval.selection import Limit, Select
from dataeval.utils.onnx import to_encoding_model
```

## Preparing an ONNX model for embeddings

Most pre-trained ONNX models output classification logits rather than embeddings. To extract embeddings, we need to
modify the model to output the features from an intermediate layer (typically before the final classification layer).

DataEval provides utility functions to help with this:

- `find_embedding_layer`: Identifies the embedding layer in a classification model
- `to_encoding_model`: Returns a modified model with the embedding layer name

We'll download a ResNet50 model and use these utilities to prepare it for embedding extraction.

```{code-cell} ipython3
def download_onnx_model(url, save_path):
    """Downloads the ONNX model if it doesn't exist locally."""
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}")
        return

    print(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")
```

```{code-cell} ipython3
# Download and prepare the model
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
model_path = "data/resnet50-v2-7.onnx"

download_onnx_model(model_url, model_path)

# Find the embedding layer and create an in-memory model that outputs it

encoding_model, embedding_layer = to_encoding_model(model_path)
print(f"Embedding layer: {embedding_layer}")
print(f"In-memory encoding model: ({len(encoding_model):,} bytes)")
```

## Loading the dataset

We'll use the VOC2012 dataset for this demonstration.

```{code-cell} ipython3
# Define transforms for ResNet50 input requirements
def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess image for ResNet50: CHW->HWC, resize, normalize, HWC->CHW."""
    hwc = image.transpose(1, 2, 0)  # Transpose to HWC
    resized = cv2.resize(hwc, (224, 224))  # Resize using standard bi-linear interpolation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std  # Normalize
    chw = normalized.transpose(2, 0, 1)  # Transpose back to CHW
    return chw
```

```{code-cell} ipython3
# Load VOC dataset
dataset = VOCDetection(root="./data", image_set="val", year="2012", download=True)
print(f"Dataset size: {len(dataset)} images")
```

## Using OnnxExtractor to generate embeddings

Now we can use the `OnnxExtractor` to generate embeddings from our dataset.

```{code-cell} ipython3
# Create the extractor with our in-memory embedding model
extractor = OnnxExtractor(
    model=encoding_model,
    transforms=preprocess,
    output_name=embedding_layer,  # Specify which output to use
)

print(extractor)
```

```{code-cell} ipython3
# Generate embeddings using the Embeddings class
# We'll use a subset for demonstration
subset = Select(dataset, Limit(100))
embeddings = Embeddings(subset, extractor=extractor, batch_size=16)

print(f"Embeddings shape: {embeddings.shape}")
```

The embeddings have shape `(N, D)` where:

- `N` is the number of images (100 in our subset)
- `D` is the embedding dimension (2048 for ResNet50)

```{code-cell} ipython3
:tags: [remove_cell]

### TEST ASSERTION CELL ###
assert embeddings.shape[0] == 100
assert embeddings.shape[1] == 2048
```
