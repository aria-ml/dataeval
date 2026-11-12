<!-- markdownlint-disable MD051 -->

# Embeddings

This page describes what embeddings are, why DataEval evaluators use them, and how to create them with DataEval's
extractors.

## What are they

An embedding is a high-dimensional vector representation of an image, produced by a feature extractor, typically a
trained neural network. Instead of working with raw pixel values, machine learning systems use embeddings as compressed
representations of image content. What an embedding preserves depends on the model, its training objective, and the
domain it was trained on.

An embedding transforms an image (millions of pixels) into a vector of typically 128 to 2048 dimensions. The
dimensionality of these vectors affects their capabilities: higher-dimensional embeddings can capture more nuanced
visual and semantic distinctions, while lower-dimensional embeddings are more efficient to compute and visualize but may
lose subtle details. With a well-matched extractor, images with similar content map to nearby vectors. How closely that
holds depends on the extractor and the data.

## Why are they important

Comparing images quantitatively requires a representation in which distance reflects similarity of content. Raw pixel
operations such as image subtraction are dominated by variations that do not change content: lighting changes, small
spatial shifts, noise, and compression artifacts can make identical scenes differ substantially. Differences in object
identity or scene context can produce smaller pixel-level changes.

A trained model maps images from a representation in which every pixel carries equal weight to one in which patterns
relevant to its training task are emphasized and other variations are suppressed. Deep networks learn features
hierarchically, from edges to textures to shapes to objects to scenes ([Bengio et al., 2013](#ref1)). Which patterns a
model emphasizes depends on its training objective.

This allows geometric operations, such as distance and neighbor search, that often align with human judgments of image
similarity. For example, embeddings of different dog photos are typically closer together in vector space than
embeddings of a dog and a car, even when lighting, pose, or background vary. This depends on the extractor and the
domain.

Embedding distances can reflect two related kinds of similarity:

**Semantic similarity** refers to images that contain similar concepts or
meanings. A photo of a golden retriever and a cartoon drawing of a dog can
have high semantic similarity despite looking visually different.

**Visual similarity** refers to images that share similar visual
characteristics like colors, textures, or shapes, regardless of semantic
content. A photo of golden wheat and a golden retriever might have high visual
similarity due to their shared color palette.

The task an embedding model was trained for affects which patterns its embeddings emphasize. Embeddings from
classification models tend to focus on whole-image features that distinguish between categories, while embeddings from
object detection models may emphasize localized features and spatial relationships. Models trained with contrastive
learning objectives ([Chen et al., 2020](#ref2)) produce embeddings optimized for similarity comparisons, while those
trained for reconstruction tasks may better preserve fine-grained visual details.

## How are they used

Several DataEval evaluators operate on embeddings, during both development and deployment phases of the ML
lifecycle. Embeddings enable geometric operations that can reveal patterns, similarities, and anomalies in image
datasets that are difficult to detect from raw pixels.

Common applications include:

- Similarity analysis: finding images with similar content or visual
  characteristics
- Clustering: grouping images by shared semantic or visual properties
- Outlier detection: identifying unusual or anomalous images in your
  dataset
- Distribution comparison: measuring how different two sets of images are
  from each other
- Duplicate detection: finding near-identical images that might indicate
  data leakage
- Drift monitoring: detecting when production data differs systematically
  from training data

DataEval can work with raw image data, but raw pixels are not recommended for these tools. DataEval treats
embeddings geometrically, as vectors in a high-dimensional space. Raw image data treated as vectors have much higher
dimensionality, and distances in high dimensions can become less discriminating ([Beyer et al., 1999](#ref3)). Their
geometric properties also do not map closely to perceptual properties. Imperceptible differences in pixel data, such as
a one-pixel shift or a slight rescaling or rotation, can produce large distances. Convert images to embeddings before
using these tools.

## Creating embeddings

Two objects create embeddings. An extractor turns a batch of images into
vectors. The {class}`.Embeddings` class applies an extractor to a dataset one
batch at a time, keeps the result, and passes it to any DataEval tool. The two steps look like this:

```python
from dataeval import Embeddings
from dataeval.extractors import TorchExtractor

extractor = TorchExtractor(model, transforms=transforms, layer_name="avgpool")
embeddings = Embeddings(dataset, extractor=extractor, batch_size=64)
```

Pass the dataset first and everything else by keyword. `Embeddings` computes the
vectors when you first use them and keeps them, so build one `Embeddings` object
for each dataset and give it to every tool that needs it. Call
`np.asarray(embeddings)` when you need a plain array.
[Create embeddings](../notebooks/h2_create_embeddings.py) runs this recipe
with PyTorch, with ONNX and with the extractors that need no model.

### Choose an extractor

| Extractor                  | Use it when                                                               | What it needs                                                    |
| -------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| {class}`.TorchExtractor`   | You have a PyTorch model, either your own or a pre-trained one            | The model, its preprocessing as `transforms`, and a `layer_name` |
| {class}`.OnnxExtractor`    | You have an ONNX model, or you want to run a model without PyTorch        | The `onnx` extra and an `output_name`                            |
| {class}`.FlattenExtractor` | You want a baseline, or the images are small, such as thumbnails          | Images of one shape. No model                                    |
| {class}`.BoVWExtractor`    | You are looking for near-duplicates that survived a rotation or a rescale | The `opencv` extra and a call to `fit` before use. No model      |

Flattened pixels are a baseline and not an assessment, because distances between
raw pixels tend to follow brightness and alignment more than content.
[Validation and trust](ValidationAndTrust.md#extractors-that-embed-the-data)
lists what each extractor embeds and how each can mislead a metric.

### Choose a model

If you have a model trained for your task, use it. Its layers have learned patterns relevant to that task, so its
embeddings are more likely to suit your domain and task.

If you do not have a trained model, choose a pre-trained one that was trained
for a task close to yours. The training objective shapes which patterns the
embeddings emphasize. The table lists model types, example models, the tasks their embeddings are commonly used for,
and the emphasis that follows from the training objective. It is a heuristic and does not rank model types for any
DataEval evaluator.

| Model type           | Example models                           | Typical tasks                                | Typical emphasis                                                                             |
| -------------------- | ---------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Image classification | ResNet, EfficientNet, Vision Transformer | BER, Class Balance, Outlier Detection        | Features that distinguish labeled categories, which can make class boundaries more separable |
| Object detection     | YOLO, R-CNN variants, DETR               | Spatial Drift, Localization Quality          | Spatial relationships and localized features, which matter when object position is relevant  |
| Self-supervised      | DINO, MAE, SimCLR                        | Drift, Prioritization, Duplicate Detection   | General visual patterns learned without class labels, used for broad visual similarity       |
| Contrastive learning | CLIP, SwAV, MoCo                         | Similarity Search, Nearest Neighbor Analysis | An objective that separates similar from dissimilar pairs, which suits comparison tasks      |
| Segmentation         | U-Net, Mask R-CNN, DeepLab               | Fine-grained Analysis, Pixel-level Quality   | Detailed spatial structure and boundaries within images                                      |

### Select the layer

A classifier ends in a layer that outputs class scores, and class scores are not
image features. Select the layer before the classifier.

- With {class}`.TorchExtractor`, name the layer in `layer_name`. Without it, the
  extractor returns the model output, which for ResNet18 is 1000 ImageNet class
  scores and not image features. `avgpool` is the layer before the classifier of
  a torchvision ResNet.
- With {class}`.OnnxExtractor`, call
  {func}`~dataeval.utils.onnx.to_encoding_model` on the exported model. It finds
  the layer before the classifier and returns a model that outputs it, together
  with the layer name to give to `output_name`.

### Put model preprocessing in the extractor

Resizing to the model's input size, normalizing with the model's mean and standard deviation, and casting the data
type adapt the images to the model. Give them to the extractor's `transforms=`, so that only the embeddings see them.
Do not put them in the {class}`.View`. Every evaluator would see them, and {class}`.Outliers` would report brightness
on normalized tensors, which are not numbers about your data. A transform that defines the dataset, such as the
resolution you deploy at, goes in the view instead.
[Image statistics](ImageStatistics.md#transforms-and-the-statistics-they-invalidate) has the rule and the statistics
each view operation invalidates.

The same rule settles a case that looks like a view operation. A model that needs three channels from a one-channel
dataset needs that conversion in `transforms=`. In the view, {class}`.SelectChannels` would state that your grayscale
dataset is an RGB dataset, and `channels` would describe the model's input layer.

### Set the batch size

`Embeddings` raises a `ValueError` when it finds no batch size in its own
argument, in the extractor or in {func}`dataeval.config.set_batch_size`. Pass
`batch_size=` every time, so the value is visible where you use it.

The value on `Embeddings` is how many images are loaded, embedded and stored in
one step. {class}`.TorchExtractor` and {class}`.OnnxExtractor` have a
`batch_size` of their own, which is how many images go through the model in one
forward pass. When both are set, the smaller value limits the forward pass. Set
the extractor value only to bound the memory of a GPU. Neither value changes the
embeddings except for differences in the last digits, which come from
floating-point rounding.

### Store large embeddings on disk

By default `Embeddings` keeps the vectors in memory. Pass `path=` to let it use a
memory-mapped file at that location when the embeddings would take more than
`memory_threshold` of the available memory. The default threshold is 0.8. When
the embeddings fit in memory, no file is written.

### Keep one row for each item

`Embeddings` stores one row for each dataset item, in order. It counts the rows
of every batch and raises an error when the extractor returns a different number
of rows, or returns a dictionary instead of an array. The count cannot show that
each row belongs to the image at the same position, so check the extractor and
not only the error.

An extractor that returns one row for each detection does not meet this
requirement. That includes {class}`.ScoresExtractor` around a detection model,
{class}`.DetectionGeometryExtractor`, and {class}`.TorchExtractor` with
`postprocess_fn`. The same holds for {class}`.ClasswiseUncertaintyExtractor`,
which returns one array for each class. {class}`.Duplicates`, {class}`.Outliers`,
{class}`.Coverage` and {class}`.Prioritize` build an `Embeddings` from the
extractor you give them, so they have the same requirement. Send these extractors
to a drift detector instead:

- Pass the extractor as the detector's `extractor=`, or call the extractor and pass the array it returns.
- For {class}`.ClasswiseUncertaintyExtractor`, call it, select the array of one class, and give that array to a
  detector. Do not pass it as `extractor=`. Fitting succeeds, because embedding is lazy, and `predict` raises a
  `TypeError`.

[Validation and trust](ValidationAndTrust.md#extractors-that-embed-a-models-predictions)
describes what each of these extractors produces.

### Give the embeddings to a tool

Each tool takes embeddings through its own argument.
[What data does each tool need?](../getting-started/input-requirements.md#how-each-tool-takes-embeddings)
lists them in one table.

## Interpretation and limitations

An embedding is specific to the extractor that produced it. Distances, neighbors, and clusters describe the geometry of
that extractor's output space and not a property of the images alone. Two extractors, or two layers of one model, give
different geometries, so results computed from different extractors are not directly comparable. A distance does not
state which image property drives it. Two images can be close for visual reasons, such as color and texture, or for
semantic reasons.

What an embedding preserves depends on the model, its training objective, and the domain it was trained on. An
extractor trained on one domain can map images from another domain to regions where distance no longer tracks the
similarity that matters for the task, because models trained on one dataset can transfer poorly to others
([Torralba & Efros, 2011](#ref4)). A tool that reads embeddings reports a property of the data as seen through the
extractor. Before interpreting a result, check that the extractor separates the content that matters in the target
domain. [Validation and trust](ValidationAndTrust.md#extractors-that-embed-the-data) lists what each extractor embeds
and how each can mislead a metric.

## Related concept pages

- [Clustering](Clustering.md): how the data are grouped in the embedding space
- [Dataset bias and coverage](DatasetBias.md): how coverage and completeness
  analyze the embedding space
- [Distribution shift](DistributionShift.md): monitoring shifts in the embedding space
- [Validation and trust](ValidationAndTrust.md#feature-extractors-a-dependency-of-both-layers):
  every extractor DataEval ships, what each one embeds, and how each can
  mislead a downstream metric
- [Acting on results](ActingOnResults.md): how to interpret and respond to
  analyses that rely on embeddings

## See this in practice

### How-to guides

- [Create embeddings](../notebooks/h2_create_embeddings.py)
- [Embed object detection crops and visualize class clusters](../notebooks/h2_embed_detection_crops.py)
- [Measure train and test dataset divergence](../notebooks/h2_measure_divergence.py)
- [Run clustering analysis](../notebooks/h2_cluster_analysis.py)

### Tutorials

- [Identify bias and correlations](../notebooks/tt_identify_bias.py):
  acting on balance and diversity findings
- [Assess an unlabeled data space](../notebooks/tt_assess_data_space.py):
  coverage gaps and embedding-space decisions
- [Monitor shifts in operational data](../notebooks/tt_monitor_shift.py):
  responding to drift detection
- [Identify out-of-distribution samples](../notebooks/tt_identify_ood_samples.py):
  responding to OOD detection

## References

1. [Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning:
   A review and new perspectives. *IEEE Transactions on Pattern Analysis
   and Machine Intelligence*, 35(8), 1798–1828. [paper](https://arxiv.org/abs/1206.5538)]{#ref1}

2. [Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework
   for contrastive learning of visual representations. *Proceedings of the 37th
   International Conference on Machine Learning (ICML)*, 1597–1607. [paper](https://arxiv.org/abs/2002.05709)]{#ref2}

3. [Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is
   "nearest neighbor" meaningful? *Proceedings of the 7th International Conference
   on Database Theory (ICDT)*, 217–235. [paper](https://link.springer.com/chapter/10.1007/3-540-49257-7_15)]{#ref3}

4. [Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. *Proceedings
   of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1521–1528. [paper](https://ieeexplore.ieee.org/document/5995347)]{#ref4}
