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
# # How to identify duplicate video

# %% [markdown]
# ## Problem statement
#
# Deduplicating a video corpus differs significantly from deduplicating static images. Videos naturally
# consist of sequential, highly correlated frames. Consequently, frame-level comparison alone is
# insufficient for identifying video-level duplication. You must analyze the relationship between
# entire sequences to answer key structural questions, including:
#
# - Is training footage present in the test split, causing data leakage?
# - Is the same video sequence stored under multiple filenames, or compressed with different codecs?
# - What portion of the video contains unique motion or information, and what portion represents static scenes?
# - Has the footage been resampled or repackaged at a different frame rate, disrupting frame-by-frame alignment?
# - Is a single object annotated under multiple track identifiers?
#
# The :class:`~dataeval.quality.Duplicates` class detects these five conditions across a multi-object tracking (MOT)
# dataset. In this guide, you will construct a sample corpus, simulate duplication scenarios, and extract the
# corresponding detection results.

# %% [markdown]
# ### When to use
#
# Use this guide when you are working with full-motion video (FMV) or any multi-object tracking (MOT) dataset, and you
# must understand the corpus composition before splitting your data or training a model. For static image datasets,
# you can refer to [How to identify duplicates](./h2_deduplicate.py), which utilizes the same class for simpler,
# image-level scenarios.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with the following packages installed:
#    - dataeval
#    - maite-datasets
# 2. A multi-object tracking dataset

# %% [markdown]
# ## Getting started

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
from collections.abc import Iterator
from typing import cast

import numpy as np
import polars as pl
from IPython.display import display
from maite_datasets.multiobject_tracking import (
    MultiobjectTrackingTargetTuple,
    SingleFrameObjectTrackingTargetTuple,
    VideoFrameTuple,
)

from dataeval.config import set_max_processes
from dataeval.data import FrameIndices, SequenceFrames
from dataeval.protocols import (
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingDatum,
    VideoFrame,
)
from dataeval.quality import Duplicates

set_max_processes(4)
pl.Config.set_tbl_width_chars(160)

# %% [markdown]
# ## Building a corpus with known duplicates
#
# Because standard FMV datasets with labeled duplicates are not readily available, you will synthesize four short
# sequences ("patrols") containing a textured ground plane with two objects tracking across it. Each scene is
# configured with a unique, low-frequency spatial layout. This design is necessary because perceptual hashing
# algorithms compress frames into low-frequency representations; consequently, two frames with identical coarse layouts
# but different high-frequency textures can produce identical hashes despite appearing distinct.

# %%
HEIGHT, WIDTH = 72, 96
YY, XX = np.mgrid[0:HEIGHT, 0:WIDTH]


def scene(seed: int, n_frames: int) -> list[np.ndarray]:
    """A textured ground plane with two objects tracking across it."""
    rng = np.random.default_rng(seed)
    terrain = rng.normal(70, 12, (HEIGHT, WIDTH))
    for _ in range(5):
        cx, cy, spread = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT), rng.uniform(150, 900)
        terrain += rng.uniform(-90, 110) * np.exp(-(((XX - cx) ** 2 + (YY - cy) ** 2) / spread))
    terrain[: rng.integers(HEIGHT // 4, 3 * HEIGHT // 4)] += rng.uniform(30, 70)

    frames = []
    for i in range(n_frames):
        image = terrain.copy()
        for k, (speed, y0, size) in enumerate(((1.4, 0.30, 90.0), (0.9, 0.68, 60.0))):
            cx = (6 + i * speed * 1.6) % (WIDTH - 12) + 6
            image += (150 - 40 * k) * np.exp(-(((XX - cx) ** 2 + (YY - HEIGHT * y0) ** 2) / size))
        frames.append(np.clip(image, 0, 255).astype(np.uint8))
    return frames


def transcode(frames: list[np.ndarray], seed: int) -> list[np.ndarray]:
    """The same footage through another codec: same content, different bytes."""
    rng = np.random.default_rng(seed)
    return [np.clip(f.astype(np.float64) + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8) for f in frames]


# %% [markdown]
# A MAITE multi-object tracking datum is a `(VideoStream, MultiobjectTrackingTarget, metadata)` tuple. The video stream
# operates as an iterable of frames rather than an indexable sequence. Because locating frame *k* requires decoding all
# preceding frames, DataEval streams video sequentially instead of indexing it directly.


# %%
def boxes_at(frame_index: int) -> SingleFrameObjectTrackingTargetTuple:
    """Two tracked objects following the same paths the pixels do."""
    corners, tracks = [], []
    for track, speed, y0 in ((0, 1.4, 0.30), (1, 0.9, 0.68)):
        cx = (6 + frame_index * speed * 1.6) % (WIDTH - 12) + 6
        corners.append([cx - 9, HEIGHT * y0 - 9, cx + 9, HEIGHT * y0 + 9])
        tracks.append(track)
    return SingleFrameObjectTrackingTargetTuple(
        boxes=np.array(corners, dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        scores=np.ones(2, dtype=np.float32),
        track_ids=np.array(tracks, dtype=np.int64),
    )


class VideoStream:
    """An iterable of decoded frames, standing in for a file a decoder would walk."""

    def __init__(self, frames: list[np.ndarray], fps: float = 30.0) -> None:
        self._frames, self._fps = frames, fps

    def __iter__(self) -> Iterator[VideoFrame]:
        for index, frame in enumerate(self._frames):
            pixels = np.stack([frame, frame, frame])
            yield VideoFrameTuple(pixels=pixels, time_s=index / self._fps, pts=index, frame_index=index)


class VideoDataset:
    """A minimal MAITE multi-object tracking dataset over in-memory footage."""

    def __init__(self, sequences: dict[str, list[np.ndarray]], dataset_id: str) -> None:
        self.metadata = DatasetMetadata({"id": dataset_id, "index2label": {0: "vehicle", 1: "person"}})
        self._data: list[MultiobjectTrackingDatum] = [
            (
                VideoStream(frames),
                MultiobjectTrackingTargetTuple(frame_tracks=[boxes_at(i) for i in range(len(frames))]),
                cast(DatumMetadata, {"id": name, "height": HEIGHT, "width": WIDTH}),
            )
            for name, frames in sequences.items()
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        return self._data[index]


# %% [markdown]
# The synthesized corpus consists of four training sequences with three distinct duplication relationships:
#
# | # | Sequence | Description |
# | --- | --- | --- |
# | 0 | `patrol_alpha` | 60 frames of baseline footage |
# | 1 | `patrol_alpha_transcode` | A transcoded version of sequence 0, simulating codec variation |
# | 2 | `patrol_bravo` | Distinct footage containing a static sequence where the camera dwells on a frame for 20 frames |
# | 3 | `patrol_charlie` | Distinct footage with no overlapping content |
#
# And one test sequence: a 20-frame clip extracted from `patrol_alpha` and transcoded. You will use this sequence to
# detect, analyze, and resolve data leakage.

# %%
alpha = scene(1, 60)
bravo = scene(5, 40)
bravo_stare = bravo[:15] + [bravo[15]] * 20 + bravo[16:]

train = VideoDataset(
    {
        "patrol_alpha": alpha,
        "patrol_alpha_transcode": transcode(alpha, seed=9),
        "patrol_bravo": bravo_stare,
        "patrol_charlie": scene(12, 45),
    },
    dataset_id="train",
)
test = VideoDataset({"eval_clip": transcode(alpha[30:50], seed=3)}, dataset_id="test")

# %% [markdown]
# ## Adjusting settings for video datasets
#
# The `hash_radius` parameter specifies the maximum Hamming distance (in bits) between two perceptual hashes for their
# corresponding frames to be classified as identical. It defaults to `0`, representing a strict, bit-for-bit match.
# While a default of `0` is appropriate for static image datasets where duplicated files (such as a re-saved PNG) are
# often bitwise identical, transcoded video frames almost always exhibit compression artifacts, noise, and other minor
# variations that prevent an exact match.
#
# If you retain the default setting for video data, the evaluation will not necessarily fail to find duplicates;
# instead, it may yield incomplete, inaccurate, or misleading matches that are difficult to detect.

# %%
strict = Duplicates().evaluate(train)
relaxed = Duplicates(hash_radius=6).evaluate(train)

for name, result in (("hash_radius=0", strict), ("hash_radius=6", relaxed)):
    row = result.sequences.data().row(0, named=True)
    print(
        f"{name}: sequences {row['item_indices']} share frames "
        f"{row['span_start'][0]}-{row['span_end'][0]}, containment {[round(c, 2) for c in row['containment']]}"
    )

# %% [markdown]
# Both configurations identify the relationship. However, the strict evaluation (`hash_radius=0`) reports that the two
# sequences share approximately half of their frames, whereas one is actually an end-to-end copy of the other. Because
# only a subset of the transcoded frames happened to produce identical hashes under strict matching, this partial result
# could be misinterpreted as a complete and accurate finding.
#
# This under-reporting also distorts the leakage measurement, where detection accuracy is critical:

# %%
for name, radius in (("hash_radius=0", 0), ("hash_radius=6", 6)):
    found = Duplicates(hash_radius=radius, min_segment_frames=10).evaluate(train, test)
    row = found.sequences.data().filter(pl.col("dataset_indices").list.n_unique() > 1).row(0, named=True)
    print(
        f"{name}: train frames {row['span_start'][0]}-{row['span_end'][0]} "
        f"== test frames {row['span_start'][1]}-{row['span_end'][1]}"
        f"   ({row['containment'][1]:.0%} of the test clip)"
    )

# %% [markdown]
# Using the default configuration, the test clip appears to be only 60% duplicated from training footage and maps to
# incorrect frames. In reality, the test clip is a 100% duplicate spanning frames 30–49 of the training sequence.
# Relying on the strict result would lead you to remove the wrong frames, leaving the majority of the data leak intact.
#
# ```{note}
# When evaluating video datasets, start with `hash_radius=6`. DataEval defines a Hamming distance of 1–5 bits as
# "highly similar" and 6–10 bits as "potentially similar." In video-hashing literature, the operational threshold is
# typically ≤30 out of 256 bits, which corresponds to approximately 10% of the hash code. The default remains `0` to
# ensure consistency with static image evaluation, as a dynamically altered default would introduce unpredictable
# behavior during execution.
#
# The `Duplicates` class automatically logs a warning when it processes video data with a `hash_radius` of `0`. You can
# enable logging to observe this warning by configuring the standard library `logging` module as described in
# [How to configure logging](./h2_configure_logging.py):
#
# ```python
# import logging
# logging.basicConfig(level=logging.WARNING)
# ```
# ```

# %% [markdown]
# ## Triaging the corpus
#
# You can begin by reviewing the high-level summary, which provides one row per sequence, lists key metrics, and
# highlights potential areas of concern.

# %%
summary = relaxed.aggregate_by_sequence()
display(summary)

# %% [markdown]
# This summary exposes two primary metrics that analyze distinct duplication characteristics:
#
# - **`redundant_fraction`**: Measures self-redundancy, representing the fraction of frames that do not introduce new
#   information relative to their immediate predecessors. In this example, sequence 2 scores the highest due to the
#   simulated 20-frame static camera dwell.
# - **`duplicate_frames` / `shared_with`**: Measures cross-sequence duplication, representing the frames that match
#   other video sequences in the corpus. Sequences 0 and 1 report 100% overlap with each other because they represent
#   identical footage, while sequences 2 and 3 share no frames with other sequences.
#
# Note that `duplicate_frames` only includes matches found in *different* sequences. Because consecutive frames in
# almost any video naturally resemble one another within standard `hash_radius` limits, treating consecutive
# self-similarity as cross-sequence duplication would incorrectly classify an entirely unique corpus as fully
# duplicated.

# %% [markdown]
# ## Identifying sequence-level duplicates
#
# The triage table indicates that sequence 0 shares content with another sequence. To determine which sequence it
# matches, quantify the overlap, and analyze the match details, you can inspect the sequence-level DataFrame.

# %%
display(
    relaxed.sequences.data().select(
        "dup_type", "item_indices", "span_start", "span_end", "containment", "mean_distance"
    )
)

# %% [markdown]
# The resulting row indicates that sequences 0 and 1 overlap from frame 0 to frame 59, with each sequence fully
# containing the other (`containment` is `[1.0, 1.0]`) at a mean Hamming distance of approximately 1 bit. This pattern
# is characteristic of a re-encoded video. To deduplicate, you can retain one sequence and discard the duplicate.
#
# The `dup_type` column categorizes the detected duplication relationships:
#
# | `dup_type` | Description |
# | --- | --- |
# | `exact` | The two sequences contain identical frames in the identical order. |
# | `segment` | The sequences overlap over a continuous interval at a fixed frame offset. |
# | `aligned` | The sequences overlap but do not advance at the same rate, indicating a speed edit (see below). |
# | `redundant` | A continuous run within a single sequence that carries no new information relative to preceding frames. |

# %% [markdown]
# ## Detecting train-test leakage
#
# Data leakage across training and test splits can artificially inflate model performance metrics. You can pass both
# splits to the `evaluate` method, and the `dataset_indices` column will specify the origin of each matching sequence.

# %%
leakage = Duplicates(hash_radius=6, min_segment_frames=10).evaluate(train, test)
leaks = leakage.crossing.aggregate_by_pair("sequence")
display(leaks)

# %% [markdown]
# The `.crossing` attribute filters the results to retain only duplication relationships that span the dataset
# boundary. The `aggregate_by_pair` method formats this data into a pair-wise view, displaying one row per matching
# pair and placing the containment metrics for each dataset in separate columns.
#
# This directional containment is critical because duplication is often asymmetric. For instance, a `containment_a` of
# `0.4` alongside a `containment_b` of `1.0` indicates that the overlapping segment comprises 40% of the training
# sequence but 100% of the test clip. Traditional symmetric similarity scores, transitive duplicate groups, and
# single-value similarity metrics cannot represent this asymmetry, often obscuring the fact that a test sequence is
# entirely leaked.
#
# The second row correctly identifies the same overlapping clip within the transcoded sequence.
#
# To retrieve the specific frame intervals, you can refer to the sequence-level rows, where frame spans are reported
# in the original source-video coordinate space:

# %%
for row in leakage.sequences.data().filter(pl.col("dataset_indices").list.n_unique() > 1).iter_rows(named=True):
    train_seq, test_seq = row["item_indices"]
    print(
        f"train[{train_seq}] frames {row['span_start'][0]}-{row['span_end'][0]}"
        f"  ==  test[{test_seq}] frames {row['span_start'][1]}-{row['span_end'][1]}"
        f"   ({row['containment'][1]:.0%} of the test clip)"
    )

# %% [markdown]
# ```{important}
# Reported spans use **source-video frame numbers**, which align with the coordinates in `unit_indices`. This ensures
# that seeking to a reported frame in a video player correctly locates the duplicate segment, regardless of any frame
# sampling rates applied during evaluation.
# ```

# %% [markdown]
# ## Identifying redundant video segments
#
# While the `redundant_fraction` provides a high-level summary of self-redundancy, you must locate the specific
# intervals to act on this information. You can filter for redundant rows, sort them by sequence length, and identify
# periods of static camera dwell.

# %%
runs = (
    relaxed
    .data()
    .filter(pl.col("dup_type") == "redundant")
    .with_columns(pl.col("unit_indices").list.len().alias("run_length"))
    .sort("run_length", descending=True)
    .select("item_indices", "unit_indices", "run_length", "mean_distance")
)
display(runs.head(4))

# %% [markdown]
# The longest sequence is 20 frames with a `mean_distance` of `0.0`, representing the static camera dwell simulated in
# sequence 2. Other redundant runs are shorter and exhibit mean Hamming distances of 2–4 bits, which typically indicates
# standard slow-motion footage rather than a static camera. You should distinguish between these two patterns: a static
# camera dwell represents redundant frames that increase storage, increase annotation costs, and add no information,
# whereas slow-motion footage continues to capture dynamic changes.
#
# This distinction is why `aggregate_by_sequence` reports the `longest_run` alongside the `redundant_fraction`. Two
# sequences can exhibit the same redundant fraction but represent entirely different physical scenarios, such as a
# single prolonged static dwell versus a scene that moves continuously but slowly:

# %%
display(summary.select("sequence", "redundant_fraction", "longest_run"))
#
# A static run of *k* frames can be reduced to a single frame without a loss of semantic content. The `redundant_frames`
# metric quantifies this potential reduction. However, you should evaluate these frames carefully before removing them:
# temporal dwell can carry valuable signal, and an object tracker trained exclusively on moving targets may fail to
# detect stationary objects.

# %% [markdown]
# ## Detecting repackaged or resampled footage
#
# Segment matching assumes a constant frame offset between sequences. If you re-export or resample a video at a
# different frame rate, the frame offset changes continuously. This causes the overlapping segment to appear as highly
# fragmented matches that fall below the minimum reporting threshold, resulting in no duplicates being detected under
# default settings.

# %%
source = scene(1, 40)
slowed = [frame for frame in source for _ in (0, 1)]
repackaged = VideoDataset({"source": source, "slowed_export": transcode(slowed, seed=4)}, dataset_id="repack")

without = Duplicates(hash_radius=6, min_segment_frames=20).evaluate(repackaged)
print(f"segments only:              {without.sequences.data().shape[0]} relation(s) found")

warped = Duplicates(hash_radius=6, min_segment_frames=20, verify_alignment=8).evaluate(repackaged)
print(f"with verify_alignment=8:    {warped.sequences.data().shape[0]} relation(s) found")

# %%
display(
    warped.sequences.data().select("dup_type", "item_indices", "span_start", "span_end", "containment", "mean_distance")
)

# %% [markdown]
# The `verify_alignment` parameter enables **dynamic time warping (DTW)**, which aligns non-linearly matching sequences
# regardless of rate variations. The resulting `aligned` row shows the baseline source sequence aligning with the
# slowed export sequence at a mean distance of slightly over one bit per frame.
#
# The value assigned to `verify_alignment` represents the maximum average Hamming distance permitted per aligned
# frame. A threshold of `8` is a recommended starting point for perceptual hashes. This alignment check is disabled by
# default because dynamic time warping has quadratic time complexity, whereas the standard segment search is
# near-linear. To maximize efficiency, DataEval only applies warping to sequence pairs that cannot be resolved by the
# segment search.

# %% [markdown]
# ## Identifying duplicated annotations
#
# At a finer granularity, you can analyze tracks—which are sequences of cropped object images tracked across
# consecutive frames. By setting `levels="track"`, you can use the same detection engine to identify duplicated
# tracks, such as a single object annotated under multiple track identifiers, a track duplicated due to a reused video
# clip, or other track-level duplication patterns.
#
# The `levels` parameter specifies the target granularities for evaluation, and any omitted levels are skipped. In this
# configuration, DataEval skips frame-level evaluation entirely and computes only track-level relationships, optimizing
# execution time.

# %%
tracked = Duplicates(hash_radius=6, min_track_frames=10).evaluate(train, levels="track")
display(
    tracked.tracks.data().select("dup_type", "item_indices", "track_indices", "span_start", "span_end", "containment")
)

# %% [markdown]
# The `track_indices` column displays the track identifiers exactly as provided in your annotations, rather than
# applying an internal renumbering scheme. This allows you to locate the flagged tracks directly in your source data. In
# this example, each row correctly pairs sequence 0 with sequence 1 because the copied sequence retains its original
# track annotations.
#
# You can configure the minimum length of a duplicated track using `min_track_frames` (which defaults to `5`), separate
# from `min_segment_frames`. While 30 frames is a typical threshold for detecting a duplicated video sequence, a much
# shorter threshold is appropriate for tracking individual objects.
#
# ```{note}
# The `levels` parameter and the `per_image`/`per_target` parameters represent alternative interfaces for configuring
# evaluation granularity. Specifying `per_target=True` is equivalent to requesting track-level relationships, as it
# prompts the package to compute crop-level hashes. You should avoid passing both `levels` and `per_image`/`per_target`
# simultaneously, as doing so will raise a configuration error rather than resolving the parameters implicitly.
# ```

# %% [markdown]
# ## Mapping results back to the dataset
#
# All detected duplicates are reported in **source-video coordinates**, specified by a sequence index and frame numbers.
# To visualize these duplicate frames, extract specific sequences, or construct a new dataset split excluding the
# duplicates, you can use the :class:`~dataeval.data.FrameIndices` class, which is compatible with `frame_sample`.
#
# For example, you can extract the leaked training sequence 0, frames 30–49:

# %%
row = leakage.crossing.sequences.data().row(0, named=True)
sequence = row["item_indices"][0]
first, last = row["span_start"][0], row["span_end"][0]

leaked = SequenceFrames(train, FrameIndices({sequence: range(first, last + 1)}))
print(f"the leaked stretch is {len(leaked)} frames of sequence {sequence}")

pixels, target, metadata = next(iter(leaked.stream()))
# The per-frame keys DataEval adds live alongside the ones MAITE declares, so read them as a dict.
meta = dict(metadata)
print(f"first of them: frame {meta['frame']} of sequence {meta['sequence']}, pixels {pixels.shape}")

# %% [markdown]
# You can use the same selector to define the **complement** of the duplicates, representing the unique footage you
# want to retain. DataEval identifies the overlapping segments, leaving the final splitting and filtering decisions to
# your discretion.

# %%
n_frames = len(train[sequence][1].frame_tracks)
keep = [frame for frame in range(n_frames) if not first <= frame <= last]
without_leak = SequenceFrames(train, FrameIndices({sequence: keep}))
print(f"sequence {sequence} without the shared stretch: {len(without_leak)} of {n_frames} frames")

# %% [markdown]
# You can apply this workflow to group members. Because `result.frames.exact` returns a list of `(sequence, frame)`
# pairs, you can aggregate them by sequence and pass the mapping to `FrameIndices`:
#
# ```python
# from collections import defaultdict
#
# wanted = defaultdict(list)
# for group in result.frames.exact:
#     for sequence, frame in group:
#         wanted[sequence].append(frame)
#
# duplicated = SequenceFrames(dataset, FrameIndices(dict(wanted)))
# ```
#
# ```{note}
# The `FrameIndices` class utilizes lazy execution; consequently, constructing a view does not trigger expensive frame
# decoding until the data is accessed, although calling `len()` remains an efficient operation. Frames are always
# returned in ascending chronological order per sequence, regardless of their specification order, because video
# streams are read sequentially forwards without rewinding.
# ```

# %% [markdown]
# ## Acting on the results
#
# The detected duplication patterns map onto several typical dataset curation decisions:
#
# | Detection Scenario | Recommended Action |
# | --- | --- |
# | `containment` near `[1.0, 1.0]` between two sequences | Identical video sequence. Discard one duplicate, and retain the sequence with higher-quality annotations. |
# | Lopsided `containment` across splits | Data leakage. Remove the overlapping frames from the test split, or partition the dataset by sequence. |
# | High `redundant_fraction` | Resample the sequence using `frame_sample` instead of deleting frames, first verifying that the static dwell does not carry meaningful signal. |
# | `aligned` relationship | Repackaged or resampled footage. Treat the aligned segment as a duplicate of the source sequence. |
# | Track-level duplicates | Annotation error. Correct the track identifiers in the labeling metadata. |
#
# When partitioning your data, you should always split **by sequence**. The `SequenceFrames` class exposes `sequence` as
# a unit-level metadata attribute. Using `split_dataset(..., split_on=["sequence"])` ensures that all frames of a given
# video sequence remain grouped on a single side of the split, which prevents you from accidentally reintroducing the
# data leakage identified during evaluation.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# hash_radius matters: the strict default under-reports the relation rather than missing it
assert strict.sequences.data().row(0, named=True)["containment"][0] < 0.6
assert relaxed.sequences.data().shape[0] == 1

# one relation, not one per diagonal
alpha_row = relaxed.sequences.data().row(0, named=True)
assert alpha_row["item_indices"] == [0, 1]
assert alpha_row["containment"] == [1.0, 1.0]

# self-repetition is not cross-sequence duplication
assert summary["duplicate_frames"].to_list() == [60, 60, 0, 0]
assert summary["shared_with"].to_list() == [1, 1, 0, 0]
assert summary["redundant_fraction"].to_list()[2] == max(summary["redundant_fraction"].to_list())

# leakage: the test clip is entirely drawn from training footage
assert leaks.shape[0] >= 1
first = leaks.row(0, named=True)
assert (first["dataset_a"], first["dataset_b"]) == (0, 1)
assert first["containment_b"] == 1.0
assert first["containment_a"] < 0.5
spans = leakage.sequences.data().filter(pl.col("dataset_indices").list.n_unique() > 1).row(0, named=True)
assert (spans["span_start"][0], spans["span_end"][0]) == (30, 49)
assert (spans["span_start"][1], spans["span_end"][1]) == (0, 19)

# the stare is the longest redundant run, and it is exact
assert runs.row(0, named=True)["run_length"] == 20
assert runs.row(0, named=True)["mean_distance"] == 0.0

# warping finds what a fixed offset cannot
assert without.sequences.data().shape[0] == 0
assert warped.sequences.data().shape[0] == 1
assert warped.sequences.data().row(0, named=True)["dup_type"] == "aligned"

# a finding leads back to the frames it names
assert len(leaked) == 20
assert meta["frame"] == 30
assert meta["sequence"] == 0
assert len(without_leak) == 40

# tracks travelled with the copied clip
assert tracked.tracks.data().shape[0] > 0
assert all(row == [0, 1] for row in tracked.tracks.data()["item_indices"].to_list())
# a level not asked for is not searched for
assert set(tracked.data()["level"].to_list()) == {"track"}

# %% [markdown]
# ## Next steps
#
# - [How to identify duplicates](./h2_deduplicate.py) — The same class on still-image datasets.
# - [Acting on Results](../concepts/ActingOnResults.md) — Strategies for addressing dataset issues found during evaluation.
# - [Data Integrity](../concepts/DataIntegrity.md) — Image-level and target-level statistics for finding data quality issues.
# - [How to build dataset views](./h2_build_dataset_views.py) — Compose filtered and transformed views over a dataset.
# - [Validation and Trust](../concepts/ValidationAndTrust.md) — Build splits that hold up, and know what your evaluation is measuring.
