# Embeddings

This page explains the role and importance of embeddings in vision tasks and
guides you through understanding how to work with them using our tools. For
curated examples using embeddings, see our
[tutorials](../tutorials/index.md).

## What are they

Embeddings are high-dimensional vector representations of images that capture
meaningful visual and semantic features in a dense numerical format. Instead
of working with raw pixel values, machine learning systems use embeddings as
compressed representations that preserve what matters most about image content.

An embedding transforms an image (millions of pixels) into a vector of
typically 128 to 2048 numbers. The dimensionality of these vectors affects
their capabilities: higher-dimensional embeddings can capture more nuanced
visual and semantic distinctions, while lower-dimensional embeddings are more
efficient to compute and visualize but may lose subtle details. These vectors
encode image content such that similar vectors represent *meaningfully* similar
images.

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

Although DataEval can work with raw image data, it is inadvisable to do so.
DataEval treats embeddings geometrically as vectors in a high-dimensional
space. Raw image data treated as vectors have much higher dimensionality, and
furthermore their geometric properties do not map cleanly to perceptual
properties. Imperceptible differences in pixel data-- from e.g. a 1 pixel
shift, or a slight rescaling or rotation--can result in large measured
distances. Therefore we strongly recommend that you first convert your images
to embeddings.

## Creating embeddings

Embeddings can be created using DataEval's {class}`.Embeddings` class. The
Embeddings class takes a neural network as input, and that network handles the
actual transformation from pixels to vectors.

If you already had a model trained for your specific task, the best embeddings
would of course come directly from that model. The {class}`.Embeddings` class
allows you to specify the appropriate layer from which to extract embeddings
during inference. These will be well-suited to your domain and task—after all,
your model has already learned exactly the patterns that matter in your data.

But you most likely won't yet have a trained model when you first want to
make embeddings. Instead, you'll choose from a set of pre-trained embedding
models, selecting ones that were trained for tasks most similar to what you
set out to accomplish. Such a neural network model will already have been
trained on large image datasets using one of various possible objectives, and
the training objective fundamentally shapes what patterns the resulting
embeddings capture. The table below shows examples of appropriate models for a
variety of metrics and tasks.

| Model Type | Example Models | Best for These Metrics | Why |
|------------|----------------|----------------------|-----|
| **Image Classification** | ResNet, EfficientNet, Vision Transformer | BER, Class Balance, Outlier Detection | Embeddings emphasize features that distinguish between labeled categories, making class boundaries clear |
| **Object Detection** | YOLO, R-CNN variants, DETR | Spatial Drift, Localization Quality | Embeddings capture spatial relationships and localized features, ideal when object position matters |
| **Self-Supervised** | DINO, MAE, SimCLR | Drift, Prioritization, Duplicate Detection | Embeddings capture general visual patterns without class bias, good for broad visual similarity |
| **Contrastive Learning** | CLIP, SwAV, MoCo | Similarity Search, Nearest Neighbor Analysis | Embeddings optimized for distinguishing similar vs. dissimilar pairs, excellent for comparison tasks |
| **Segmentation** | U-Net, Mask R-CNN, DeepLab | Fine-grained Analysis, Pixel-level Quality | Embeddings understand detailed spatial structure and boundaries within images |
