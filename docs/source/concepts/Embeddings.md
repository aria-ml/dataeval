# Embeddings

This page explains the role and importance of embeddings in vision tasks and
guides you through understanding how to work with them using our tools.

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

Two objects create embeddings. An extractor turns a batch of images into
vectors. The {class}`.Embeddings` class applies an extractor to a dataset one
batch at a time, keeps the result, and passes it to any DataEval tool. Every
example in the documentation uses the same two steps:

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
[How to create embeddings](../notebooks/h2_encode_with_onnx.py) runs this recipe
with PyTorch, with ONNX and with the extractors that need no model.

### Choose an extractor

| Extractor                  | Use it when                                                               | What it needs                                                    |
| -------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| {class}`.TorchExtractor`   | You have a PyTorch model, either your own or a pre-trained one            | The model, its preprocessing as `transforms`, and a `layer_name` |
| {class}`.OnnxExtractor`    | You have an ONNX model, or you want to run a model without PyTorch        | The `onnx` extra and an `output_name`                            |
| {class}`.FlattenExtractor` | You want a baseline, or the images are small, such as MNIST               | Images of one shape. No model                                    |
| {class}`.BoVWExtractor`    | You are looking for near-duplicates that survived a rotation or a rescale | The `opencv` extra and a call to `fit` before use. No model      |

Flattened pixels are a baseline and not an assessment, because distances between
raw pixels follow brightness and alignment more than content.
[Validation and Trust](ValidationAndTrust.md#extractors-that-embed-the-data)
lists what each extractor embeds and how each can mislead a metric.

### Choose a model

If you have a model trained for your task, use it. Its layers have already
learned the patterns that matter in your data, so its embeddings suit your domain
and task.

If you do not have a trained model, choose a pre-trained one that was trained
for a task close to yours. The training objective shapes which patterns the
embeddings capture. The table shows examples of models for a range of metrics
and tasks.

| Model Type               | Example Models                           | Best for These Metrics                       | Why                                                                                                      |
| ------------------------ | ---------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Image Classification** | ResNet, EfficientNet, Vision Transformer | BER, Class Balance, Outlier Detection        | Embeddings emphasize features that distinguish between labeled categories, making class boundaries clear |
| **Object Detection**     | YOLO, R-CNN variants, DETR               | Spatial Drift, Localization Quality          | Embeddings capture spatial relationships and localized features, ideal when object position matters      |
| **Self-Supervised**      | DINO, MAE, SimCLR                        | Drift, Prioritization, Duplicate Detection   | Embeddings capture general visual patterns without class bias, good for broad visual similarity          |
| **Contrastive Learning** | CLIP, SwAV, MoCo                         | Similarity Search, Nearest Neighbor Analysis | Embeddings optimized for distinguishing similar vs. dissimilar pairs, excellent for comparison tasks     |
| **Segmentation**         | U-Net, Mask R-CNN, DeepLab               | Fine-grained Analysis, Pixel-level Quality   | Embeddings understand detailed spatial structure and boundaries within images                            |

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

### Set the batch size

`Embeddings` raises a `ValueError` when it finds no batch size in its own
argument, in the extractor or in {func}`dataeval.config.set_batch_size`. Pass
`batch_size=` every time, so the value is visible where you use it.

The value on `Embeddings` is how many images are loaded, encoded and stored in
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

- Pass the extractor as the detector's `extractor=`, or call the extractor and
  pass the array it returns.
- For {class}`.ClasswiseUncertaintyExtractor`, call it, select the array of one
  class, and give that array to a detector. Do not pass it as `extractor=`.
  Fitting succeeds, because encoding is lazy, and `predict` raises a `TypeError`.

[Validation and Trust](ValidationAndTrust.md#extractors-that-embed-a-models-predictions)
describes what each of these extractors produces.

### Give the embeddings to a tool

Each tool takes embeddings through its own argument.
[What data does each tool need?](../getting-started/input-requirements.md#how-each-tool-takes-embeddings)
lists them in one table.

## Related concept pages

- [Clustering](Clustering.md) — how the data are grouped in the embedding space
- [Dataset Bias and Coverage](DatasetBias.md) — how coverage and completeness
  analyze the embedding space
- [Distribution Shift](DistributionShift.md) — monitoring shifts in the embedding space
- [Validation and Trust](ValidationAndTrust.md#feature-extractors-a-dependency-of-both-layers)
  — every extractor DataEval ships, what each one embeds, and how each can
  mislead a downstream metric
- [Acting on Results](ActingOnResults.md) — how to interpret and respond to
  analyses that rely on embeddings

## See this in practice

### How-to guides

- [How to create embeddings](../notebooks/h2_encode_with_onnx.py)
- [Embed object detection crops and visualize class clusters](../notebooks/h2_embed_detection_crops.py)
- [How to measure train and test dataset divergence](../notebooks/h2_measure_divergence.py)
- [How to perform cluster analysis](../notebooks/h2_cluster_analysis.py)

### Tutorials

- [Identify bias and correlations tutorial](../notebooks/tt_identify_bias.py) —
  acting on balance and diversity findings
- [Assessing the data space tutorial](../notebooks/tt_assess_data_space.py) —
  coverage gaps and embedding-space decisions
- [Monitoring distribution shift tutorial](../notebooks/tt_monitor_shift.py) —
  responding to drift detection
- [Identifying OOD samples tutorial](../notebooks/tt_identify_ood_samples.py) —
  responding to OOD detection

## References

1. Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning:
   A review and new perspectives. *IEEE Transactions on Pattern Analysis
   and Machine Intelligence*, 35(8), 1798–1828. [paper](https://arxiv.org/abs/1206.5538)

2. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework
   for contrastive learning of visual representations. *Proceedings of the 37th
   International Conference on Machine Learning (ICML)*, 1597–1607. [paper](https://arxiv.org/abs/2002.05709)

3. Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is
   "nearest neighbor" meaningful? *Proceedings of the 7th International Conference
   on Database Theory (ICDT)*, 217–235. [paper](https://link.springer.com/chapter/10.1007/3-540-49257-7_15)

4. Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. *Proceedings
   of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1521–1528. [paper](https://ieeexplore.ieee.org/document/5995347)
