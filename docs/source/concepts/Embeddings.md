# Embeddings

This page explains the role and importance of embeddings in vision tasks and
guides you through understanding how to work with them using our tools. For
implementation details, see our [tutorials](../tutorials/index.md).

## What are they

Embeddings are high-dimensional vector representations of images that capture
meaningful visual and semantic features in a dense, numerical format. Instead
of working with raw pixel values, machine learning systems use embeddings as
compressed representations that preserve what matters most about image content.

An embedding transforms an image (millions of pixels) into a vector of
typically 128 to 2048 numbers. The dimensionality of these vectors affects
their capabilities: higher-dimensional embeddings can capture more nuanced
visual and semantic distinctions, while lower-dimensional embeddings are more
efficient to compute and visualize but may lose subtle details. These vectors
encode what the image contains in a way that similar vectors represent
meaningfully similar images.

## Why are they important

Embeddings solve a fundamental problem in computer vision: how do you
quantitatively compare images in a meaningful way? Raw pixel operations like
image subtraction are dominated by irrelevant variations—lighting changes,
small spatial shifts, noise, and compression artifacts can make identical
scenes appear completely different computationally. Meanwhile, genuinely
important differences like object identity or scene context might produce
smaller pixel-level changes.

Embeddings transform images from a representation where all visual information
has equal importance to one where task-relevant visual and semantic patterns
are emphasized and irrelevant variations are suppressed. They learn to extract
and combine visual features hierarchically—from edges to textures to shapes to
objects to scenes—in ways that capture what actually matters for understanding
image content.

This transformation enables geometric operations that align with human
intuition about image similarity. For example, embeddings of different dog
photos will be closer together in vector space than embeddings of dogs and
cars, regardless of variations in lighting, pose, or background.

Embeddings capture two distinct but related types of similarity:

**Semantic similarity** refers to images that contain similar concepts or
meanings. A photo of a golden retriever and a cartoon drawing of a dog would
have high semantic similarity despite looking visually different.

**Visual similarity** refers to images that share similar visual
characteristics like colors, textures, or shapes, regardless of semantic
content. A photo of golden wheat and a golden retriever might have high visual
similarity due to their shared color palette.

The type of task an embedding model was trained for significantly affects what
patterns they emphasize. Embeddings from classification models tend to focus on
whole-image features that distinguish between categories, while embeddings from
object detection models may emphasize localized features and spatial
relationships. Models trained with contrastive learning objectives create
embeddings optimized for similarity comparisons, while those trained for
reconstruction tasks may better preserve fine-grained visual details.

## How are they used

Embeddings serve as the foundation for DataEval's analysis capabilities during
both development and deployment phases of your ML lifecycle. The vector
representations enable geometric operations that reveal patterns, similarities,
and anomalies in your image datasets that would be impossible to detect from
raw pixels.

Common applications include:

- **Similarity analysis**: Finding images with similar content or visual
characteristics
- **Clustering**: Grouping images by shared semantic or visual properties
- **Outlier detection**: Identifying unusual or anomalous images in your
dataset
- **Distribution comparison**: Measuring how different two sets of images are
from each other
- **Duplicate detection**: Finding near-identical images that might indicate
data leakage
- **Drift monitoring**: Detecting when production data differs systematically
from training data

DataEval cannot work directly with raw image data. You must first convert your
images to embeddings using DataEval's {class}`.Embeddings` class, which handles
the transformation from pixels to vectors.

## Creating embeddings

Embeddings are created by neural network models trained on large image
datasets using various objectives. The training objective fundamentally shapes
what patterns the resulting embeddings capture. Models trained with supervised
classification learn to emphasize features that distinguish between labeled
categories. Contrastive learning approaches train models to recognize similar
and dissimilar image pairs, creating embeddings optimized for similarity
comparisons. Self-supervised methods learn representations by solving tasks
like image reconstruction.

DataEval supports custom embedding models that you can train for your
specific domain. When choosing an embedding model, consider what task it was
originally trained for. Classification-based embeddings excel at capturing
category-level distinctions, while detection-based embeddings may better
represent spatial relationships and localized features.

For object detection scenarios, additional considerations apply. When target
objects are small relative to the overall image, their visual information can
be overwhelmed by background textures and context. In these cases, specialized
training approaches that emphasize object-level features may be necessary to
create embeddings where small targets remain detectable and distinguishable.

Understanding your model's training objective helps predict how it will
represent your images and what types of patterns DataEval's analysis tools
will be able to detect. Different analysis tasks may benefit from different
embedding approaches—some metrics require embeddings that clearly separate
different classes, while others work well with embeddings that primarily
capture visual similarity. Two concrete examples: {doc}`BER` should be trained
with a classification objective so that class overlap can be evaluated, while
{doc}`Drift` or {doc}`Prioritization` may perform well enough using a
self-supervised embedding that captures only visual features.
