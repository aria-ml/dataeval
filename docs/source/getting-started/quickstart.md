# Getting Started

DataEval helps you evaluate image datasets for quality, bias, scope, distribution
shift, and performance limits. It implements [Modular AI Trustworthy Engineering
(MAITE)](https://mit-ll-ai-technology.github.io/maite/)-compliant metrics that
integrate with the broader [Joint AI T&E Infrastructure Capability (JATIC)](https://cdao.pages.jatic.net/public/)
suite of tools.

:::{note}
DataEval imposes no restrictions on image type. It accepts any image modality
(RGB, IR, EO, multispectral, greyscale, and others) at any bit depth (8-bit, 16-bit, 32-bit, etc.)
and channel count (1+).
:::

:::{important}
Some DataEval functions and classes apply only to image classification tasks, while
others apply only to object detection tasks. For more information regarding when to use
each function or class see the [Functional Overivew page](../reference/FunctionalOverview.md)
for details.
:::

---

## Step 1: Install DataEval

DataEval requires Python 3.10 or higher. It has been tested on Ubuntu and Windows.
macOS users may encounter platform-specific issues; report these via the issue
tracker.

:::::{tab-set}

::::{tab-item} pip
DataEval can be installed via `pip` from [PyPI](https://pypi.org/project/dataeval/):

```bash
pip install dataeval
```

::::

::::{tab-item} conda-forge
DataEval can be installed via `conda` from
[conda-forge](https://github.com/conda-forge/dataeval-feedstock):

```bash
conda install -c conda-forge dataeval
```

::::
:::::

:::{seealso}
For details on optional extras, installing from source, or developer setup,
see [Installation](installation.md).
:::

---

## Step 2: Prepare your dataset

DataEval has two input paths depending on which part of the library you are using.

**`dataeval.core`** provides stateless functions that operate directly on NumPy
arrays — embeddings, labels, image hashes, and statistics. No dataset object is
required. Call these functions with arrays and get results back directly. Examples
include {func}`.compute_stats`, {func}`.label_errors`, {func}`.divergence_mst`,
and {func}`.ber_knn`.

**`dataeval.quality`, `dataeval.bias`, `dataeval.scope`, `dataeval.shift`, and `dataeval.performance`**
provide stateful evaluator classes ({class}`.Duplicates`, {class}`.Outliers`,
{class}`.Prioritize`, {class}`.Balance`, drift detectors, and so on). These
accept [MAITE](https://mit-ll-ai-technology.github.io/maite/)-compliant
datasets, {class}`.Metadata` or {class}`.Embeddings` depending on the evaluator.

If your data is not yet in MAITE format, the sections below show what is
required and how to wrap a common format, for both image classification and
object detection tasks.

### Image classification dataset

A MAITE-compliant image classification dataset implements `__len__` and
`__getitem__`, where each item is a tuple of `(image, label, metadata)`.
Images must be NumPy arrays of shape `(H, W, C)`. Labels must be one-hot
encoded arrays of shape `(num_classes,)`. Metadata must be a `DatumMetadata`
object with at minimum an `id` field.

```python
import maite.protocols as mp
import maite.protocols.image_classification as ic
import numpy as np


class MyImageClassificationDataset(ic.Dataset):
    metadata: mp.DatasetMetadata

    def __init__(self, images: list[np.ndarray], labels: list[int], num_classes: int) -> None:
        # images: list of np.ndarray, each shape (H, W, C)
        # labels: list of int (class indices)
        self._images = images
        self._labels = labels
        self._num_classes = num_classes

        self.metadata = mp.DatasetMetadata(
            id="my_image_classification_dataset",
            index2label={i: f"class_{i}" for i in np.unique(labels)},  # example mapping
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[ic.InputType, ic.TargetType, ic.DatumMetadataType]:
        return (
            self._images[idx],  # np.ndarray (H, W, C)
            np.eye(self._num_classes, dtype=np.float32)[self._labels[idx]],  # np.ndarray (num_classes,)
            ic.DatumMetadataType(id=idx),
        )
```

### Object detection dataset

A MAITE-compliant object detection dataset follows the same three-tuple
structure, but the label element is replaced by a detection target object
carrying per-box labels, bounding boxes, and scores. Bounding boxes use
`(x0, y0, x1, y1)` format. Labels and scores are per-box, not per-image.

```python
import maite.protocols as mp
import maite.protocols.object_detection as od
import numpy as np


class DetectionTarget(od.TargetType):
    """Holds per-box labels, boxes, and one-hot scores for one image."""

    def __init__(self, labels: list[int], boxes: list[list[float]], num_classes: int):
        # labels: list of int, one per box
        # boxes:  list of [x0, y0, x1, y1], one per box
        self._labels = labels
        self._boxes = boxes
        self._scores = np.eye(num_classes)[labels]

    @property
    def labels(self) -> mp.ArrayLike:
        return self._labels

    @property
    def boxes(self) -> mp.ArrayLike:
        return self._boxes

    @property
    def scores(self) -> mp.ArrayLike:
        return self._scores


class MyObjectDetectionDataset(od.Dataset):
    def __init__(
        self, images: list[np.ndarray], labels: list[list[int]], boxes: list[list[list[float]]], num_classes: int
    ) -> None:
        # images: list of np.ndarray, each shape (H, W, C)
        # labels: list of list[int] — per-box class indices, one list per image
        # boxes:  list of list[[x0,y0,x1,y1]] — one list per image
        self._images = images
        self._labels = labels
        self._boxes = boxes
        self._num_classes = num_classes

        self.metadata = mp.DatasetMetadata(
            id="my_object_detection_dataset",
            index2label={i: f"class_{i}" for i in np.unique(labels)},  # example mapping
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[od.InputType, od.TargetType, od.DatumMetadataType]:
        return (
            self._images[idx],  # np.ndarray (H, W, C)
            DetectionTarget(self._labels[idx], self._boxes[idx], self._num_classes),
            od.DatumMetadataType(id=idx),
        )
```

### Wrapping a PyTorch dataset

If your data is in a PyTorch `Dataset`, wrap it to conform to the MAITE
protocol. Note that `torchvision` tensors are `(C, H, W)` — permute to
`(H, W, C)` before passing to DataEval.

```python
import maite.protocols as mp
import maite.protocols.image_classification as ic
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

tv_cifar10 = CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())


class MyCIFAR10Wrapper(ic.Dataset):
    def __init__(self, source: CIFAR10) -> None:
        self._source = source
        self.metadata = mp.DatasetMetadata(
            id="tv_cifar10",
            index2label={
                0: "airplane",
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck",
            },
        )

    def __len__(self) -> int:
        return len(tv_cifar10)

    def __getitem__(self, idx: int) -> tuple[ic.InputType, ic.TargetType, ic.DatumMetadataType]:
        tv_datum: tuple[torch.Tensor, int] = tv_cifar10[idx]
        image = tv_datum[0].permute(1, 2, 0).numpy()  # Permute image from (C, H, W) to (H, W, C)
        label = np.eye(10, dtype=np.float32)[tv_datum[1]]  # Convert label to one-hot encoding
        return image, label, mp.DatumMetadata(id=idx)


dataset: ic.Dataset = MyCIFAR10Wrapper(tv_cifar10)
```

---

## Step 3: Run your first evaluation

The example below uses {class}`.Duplicates` from `dataeval.quality` to detect
near-duplicate images by finding groups of embeddings that are similar in
embedding space. Duplicates inflate benchmark scores and cause models to overfit
to repeated collection events rather than generalizing to new conditions.

```python
from torch.nn import Flatten

from dataeval.extractors import TorchExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates

# Configure a feature extractor using a pre-trained PyTorch model.
# Here we use a simple Flatten layer for demonstration, but in practice
# you would use a more powerful model like a pre-trained ResNet or ViT.
extractor = TorchExtractor(Flatten())

# Find near-duplicates using only embedding-based clustering.
# An aggressive cluster_threshold of 1.5 should produce detections
# of near duplicates even with a simple Flatten extractor.
evaluator = Duplicates(
    flags=ImageStats.NONE,
    cluster_algorithm="hdbscan",
    cluster_threshold=1.5,
    extractor=extractor,
    batch_size=64,
)
result = evaluator.evaluate(dataset)

# Near duplicates are grouped into sets of indices that are within
# the specified cluster_threshold in embedding space.
print(result)
```

```text
shape: (3, 5)
┌──────────┬───────┬──────────┬────────────────┬─────────────┐
│ group_id ┆ level ┆ dup_type ┆ item_indices   ┆ methods     │
│ ---      ┆ ---   ┆ ---      ┆ ---            ┆ ---         │
│ i64      ┆ str   ┆ str      ┆ list[i64]      ┆ list[str]   │
╞══════════╪═══════╪══════════╪════════════════╪═════════════╡
│ 0        ┆ item  ┆ near     ┆ [18586, 39942] ┆ ["cluster"] │
│ 1        ┆ item  ┆ near     ┆ [23157, 31426] ┆ ["cluster"] │
│ 2        ┆ item  ┆ near     ┆ [32024, 49135] ┆ ["cluster"] │
└──────────┴───────┴──────────┴────────────────┴─────────────┘
```

A result with many large groups is a signal that your dataset contains
repeated collection events. Before training, remove all but one sample from
each group. See the [deduplication how-to guide](../notebooks/h2_deduplicate.md)
for a complete walkthrough, including how to choose which sample to keep.

---

## Where to go next

Not sure what to evaluate first? Use the [Which tool should I use?](which-tool.md)
guide to find the right evaluator for your situation.

Know which tool to use, then check out the [Functional Overview](../reference/FunctionalOverview.md)
for a quick-reference table of each algorithm's inputs, outputs, and task applicability.

If you prefer to learn by doing, start with the
[data cleaning tutorial](../notebooks/tt_clean_dataset.md). It walks through
the most common first-pass analysis tasks — duplicates, outliers, and image
quality — using a realistic dataset.
