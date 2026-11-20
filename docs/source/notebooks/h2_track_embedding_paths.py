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
# # Measure the paths of tracks in embedding space
#
# A tracked object is seen in many frames, so the embeddings of its crops form a path rather than one point. The
# experimental {func}`.track_path_stats` measures how far each track's path travels, the distance its crops would give
# in a random order, and the direction of travel. See [Embeddings](../concepts/Embeddings.md), and the function's row
# in [Validation evidence by evaluator](../reference/ValidationEvidence.md) for what its tests cover.
#
# Use it when you want to see which tracks of a tracking dataset change most between their first frames and their last.
# It does not detect identity switches.
#
# You need `dataeval`, `dataeval-plots`, `maite-datasets` and `torchvision`.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import numpy as np
import polars as pl
import torch
from IPython.display import display
from maite_datasets.multiobject_tracking import VisDroneMOT
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.config import set_device
from dataeval.core import track_path_stats
from dataeval.data import DetectionCrops, Indices, SequenceFrames, View
from dataeval.extractors import TorchExtractor

set_device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Load one video as frames
#
# Load the validation videos of VisDrone2019-MOT, the drone tracking benchmark of
# [Zhu et al. (2021)](https://arxiv.org/abs/2001.06303), through the
# [Voxel51 copy](https://huggingface.co/datasets/Voxel51/visdrone-mot), shared under the
# [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license, which downloads about 1.7 GB on the
# first run. Keep one video, as frames with {class}`.SequenceFrames`.

# %%
dataset = VisDroneMOT(root="./data", image_set="val", download=True)
frames = SequenceFrames(View(dataset, Indices([3])))
print(f"{frames[0][2]['source_id']}: {len(frames)} frames")

# %% [markdown]
# ## Crop each detection
#
# Crop each box with `square="pad"`, which keeps the object's pixels and fills the rest of the square, and with
# `min_size=16`, which drops boxes whose shorter side is under 16 pixels, so those detections are missing from their
# tracks. `crops.track_ids` gives each crop's track, and `frames.frame_map` its video and frame.

# %%
crops = DetectionCrops(frames, min_size=16, square="pad")
video, frame = frames.frame_map[crops.item_indices].T
print(f"{len(crops)} crops, {crops.n_dropped} dropped, {len(np.unique(crops.track_ids))} tracks")

# %% [markdown]
# ## Embed the crops
#
# Embed each crop with a pretrained ResNet18, as
# [Embed object detection crops and visualize class clusters](./h2_embed_detection_crops.py) shows. It is an ImageNet
# classifier, not a model trained on drone footage.

# %%
weights = ResNet18_Weights.DEFAULT
extractor = TorchExtractor(
    resnet18(weights=weights, progress=False), transforms=weights.transforms(), layer_name="avgpool"
)
embeddings = Embeddings(crops, extractor=extractor, batch_size=64)

# %% [markdown]
# ## Measure each track's path
#
# Each embedding is scaled to unit length, and `path_displacement` is the distance, from 0 to 2, between the mean
# embedding of the first third of a track's crops and that of the last third. Compare `path_displacement` between
# tracks, and not its ratio to `path_displacement_null`, the root-mean-square of the same distance over every order of
# the crops, because the ratio grows with the number of crops in a track.

# %%
result = track_path_stats(embeddings, crops.track_ids, frame, video)
paths = pl.DataFrame(result["stats"]).drop_nans().sort("path_displacement", descending=True)
display(paths)

# %% [markdown]
# The `UserWarning` counts the 56 of 155 tracks with fewer than 60 crops, the default `min_points`, and those are NaN.
# The table lists the other 99 farthest first, from about 0.70 down to 0.22, so its top rows are the tracks whose
# embeddings move most between their first third and their last.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(frames) == 363
assert frames[0][2]["source_id"] == "uav0000182_00000_v"
assert (len(crops), crops.n_dropped) == (17107, 564)
assert (crops.track_ids >= 0).all()  # so np.unique counts tracks only
assert len(np.unique(crops.track_ids)) == len(result["stats"]["track_ids"]) == 155
assert paths.height == 99
assert int(np.isnan(result["stats"]["path_displacement"]).sum()) == 56
displacement = paths["path_displacement"].to_numpy()
assert paths["n_points"].to_numpy().min() >= 60
assert abs(displacement.min() - 0.22) < 0.02
assert abs(displacement.max() - 0.70) < 0.02
assert (np.diff(displacement) <= 0).all()  # "farthest first"
assert displacement.max() <= 2.0  # "from 0 to 2"

# %% [markdown]
# ## Look at the farthest and nearest tracks
#
# Trace the first and last tracks of the table back to their crops with `crops.track_ids`, in frame order, and show six
# crops of each. The animation plays 60 crops of each, the farthest on the left, resized to one size because a GIF has
# one frame size.

# %%
farthest, nearest = paths["track_ids"][0], paths["track_ids"][-1]


def crops_of(track: int, n: int) -> np.ndarray:
    """`n` crop rows of one track, evenly spaced from its first frame to its last."""
    rows = np.flatnonzero(crops.track_ids == track)
    rows = rows[np.argsort(frame[rows])]
    return rows[np.linspace(0, len(rows) - 1, n).round().astype(int)]


shown = np.concatenate([crops_of(farthest, 6), crops_of(nearest, 6)])
dep.plot(
    crops,
    indices=shown.tolist(),
    images_per_row=6,
    titles=[f"track {crops.track_ids[row]}, frame {frame[row]}" for row in shown],
    figsize=(12, 4.6),
)


# %%
def square(row: int, side: int = 128) -> np.ndarray:
    """One crop, resized to `side` x `side` pixels."""
    return np.asarray(Image.fromarray(np.asarray(crops[int(row)][0]).transpose(1, 2, 0)).resize((side, side)))


pair = [np.hstack([square(a), square(b)]) for a, b in zip(crops_of(farthest, 60), crops_of(nearest, 60), strict=True)]
display(dep.animate(pair, boxes=False, fps=8))

# %% [markdown]
# The farthest track, a loaded tricycle, is seen from changing angles, a sign covers part of it in its early crops and
# its last crop is blurred, while the nearest, a motorbike, looks much the same in most of its crops. A direction
# describes the path and does not state its cause: the object's pose and scale, what covers it, the pixels inside its
# box and the camera can each move an embedding.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (farthest, nearest) == (86, 49)  # the two tracks the text describes
label_of = {int(crops.track_ids[row]): int(np.argmax(np.asarray(crops[int(row)][1]))) for row in shown}
assert dataset.index2label[label_of[86]] == "tricycle"
assert dataset.index2label[label_of[49]] == "motor"
assert [frame[row] for row in crops_of(farthest, 6)] == [0, 63, 126, 190, 253, 317]
assert len(pair) == 60
assert pair[0].shape == (128, 256, 3)

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Feature vectors extracted per crop for evaluation and analysis.
# - [Embed object detection crops and visualize class clusters](./h2_embed_detection_crops.py) — Embed the boxes of a
#   detection dataset and see how the classes cluster.
# - [Measure frames and tracks in a tracking dataset](./h2_measure_frames_and_tracks.py) — Measure frame timing, exact
#   repeats and track steps.
