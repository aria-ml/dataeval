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
# Deduplicating video is not the same as deduplicating images. The frames of a video are sequential and highly
# correlated, so comparing frames alone cannot tell you whether two videos are the same video. You have to compare
# whole sequences to answer questions such as:
#
# - Is training footage also sitting in the test split?
# - Is the same footage stored twice, under two filenames or two codecs?
# - How much of a sequence carries new information, and how much of it is static?
# - Was the footage resampled or repackaged at another frame rate, so the frames no longer line up one to one?
# - Is one object annotated twice, under two track identifiers?
#
# The :class:`~dataeval.quality.Duplicates` class answers these questions for a multi-object tracking (MOT) dataset.
# In this guide you will build a small corpus with known duplicates, run each detection against it, and read the
# results.

# %% [markdown]
# ### When to use
#
# Use this guide when you work with full-motion video (FMV) or any multi-object tracking (MOT) dataset and you need to
# know what the corpus contains before you split it or train on it. For still images, use
# [How to identify duplicates](./h2_deduplicate.py), which covers the same class on simpler, image-level cases.

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
# FMV datasets with labeled duplicates are not readily available, so you will synthesize four short sequences. Each
# one is a textured ground plane with two objects moving across it, and each one is given a different coarse layout.
#
# The layouts differ because of how perceptual hashing works. The hash resizes a frame, runs a discrete cosine
# transform over it, and keeps only the lowest-frequency coefficients. What survives is the coarse layout of the
# frame. Everything finer than that is discarded, which covers sensor noise and compression artifacts but also grain,
# fine texture and small detail. That is what makes the hash useful here, since a re-encoded frame hashes to the same
# value as its source and you are left comparing content rather than bytes. It also means two frames that share a
# coarse layout hash alike even when their fine detail differs, so distinct scenes need distinct layouts.

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
    """The same footage re-encoded: the same content, different pixel values."""
    rng = np.random.default_rng(seed)
    return [np.clip(f.astype(np.float64) + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8) for f in frames]


# %% [markdown]
# A MAITE multi-object tracking datum is a `(VideoStream, MultiobjectTrackingTarget, metadata)` tuple. The video
# stream is an iterable of frames rather than an indexable sequence. Reaching frame *k* means decoding every frame
# before it, so DataEval streams video sequentially instead of indexing into it.


# %%
def boxes_at(frame_index: int) -> SingleFrameObjectTrackingTargetTuple:
    """Two tracked objects, boxed where the frame index puts them."""
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
    """An iterable of decoded frames, in place of the file a decoder would read."""

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
# The corpus holds four training sequences with three kinds of duplication built into them:
#
# | # | Sequence | Contents |
# | --- | --- | --- |
# | 0 | `sequence_a` | 60 frames of baseline footage |
# | 1 | `sequence_a_transcode` | Sequence 0 re-encoded: the same footage, different pixel values |
# | 2 | `sequence_b` | Different footage, holding on one frame for 20 frames |
# | 3 | `sequence_c` | Different footage that reuses frames 5-24 of sequence 0 as its own frames 15-34 |
#
# It also holds one test sequence: a 20-frame clip cut from `sequence_a` and re-encoded. You will use that sequence
# to find, measure and remove a data leak.
#
# Sequences 0 and 1 duplicate each other end to end. Sequence 3 covers the case where part of one sequence duplicates
# part of another, which the class reports the same way whether the two sequences sit in one split or across two.

# %%
frames_a = scene(1, 60)
frames_b = scene(5, 40)
frames_b_dwell = frames_b[:15] + [frames_b[15]] * 20 + frames_b[16:]
frames_c = scene(12, 45)
frames_c_reuse = frames_c[:15] + transcode(frames_a[5:25], seed=7) + frames_c[15:30]

train = VideoDataset(
    {
        "sequence_a": frames_a,
        "sequence_a_transcode": transcode(frames_a, seed=9),
        "sequence_b": frames_b_dwell,
        "sequence_c": frames_c_reuse,
    },
    dataset_id="train",
)
test = VideoDataset({"test_clip": transcode(frames_a[30:50], seed=3)}, dataset_id="test")

# %% [markdown]
# ## Adjusting settings for video datasets
#
# `hash_radius` is the largest Hamming distance, in bits, at which two frame hashes still count as the same frame. It
# defaults to `0`, an exact match across all 64 bits of the hash. That default suits still images, where a duplicated
# file is often byte for byte identical. Re-encoded video is not: compression artifacts and noise move a bit or two
# in most frames, and only some frames survive an exact match.
#
# Leaving the default in place does not make the evaluation miss video duplicates outright, and it does not report
# relations that are not there. It finds fewer of the frames that make up each relation, so the spans and containment
# it reports understate the relation it did find.
#
# The sequences here are short, so the code below also lowers `min_segment_frames` from its default of `30` to `10`.
# Otherwise the 20-frame stretch that sequence 3 reuses falls below the reporting threshold.

# %%
strict = Duplicates(min_segment_frames=10).evaluate(train)
relaxed = Duplicates(hash_radius=6, min_segment_frames=10).evaluate(train)

for name, result in (("hash_radius=0", strict), ("hash_radius=6", relaxed)):
    row = result.sequences.data().filter(pl.col("item_indices") == [0, 1]).row(0, named=True)
    print(
        f"{name}: sequences {row['item_indices']} share frames "
        f"{row['span_start'][0]}-{row['span_end'][0]}, containment {[round(c, 2) for c in row['containment']]}"
    )

# %% [markdown]
# Both settings find the relationship between sequences 0 and 1. Under `hash_radius=0`, roughly half the frames match
# and the pair reads as a partial overlap, because only some of the re-encoded frames happened to hash identically.
# Sequence 1 is a copy of sequence 0 from end to end, which is what `hash_radius=6` reports.
#
# A partial overlap is a valid finding in its own right, and sequence 3 is one. The problem here is the extent. Read
# the strict result as it stands and you would keep a sequence that is a full copy of one you already have.
#
# The same under-reporting shrinks a leakage measurement, where the extent is what you act on:

# %%
extent = {}
for name, radius in (("hash_radius=0", 0), ("hash_radius=6", 6)):
    found = Duplicates(hash_radius=radius, min_segment_frames=10).evaluate(train, test)
    row = found.sequences.data().filter(pl.col("dataset_indices").list.n_unique() > 1).row(0, named=True)
    extent[radius] = row
    print(
        f"{name}: train frames {row['span_start'][0]}-{row['span_end'][0]} "
        f"== test frames {row['span_start'][1]}-{row['span_end'][1]}"
        f"   ({row['containment'][1]:.0%} of the test clip)"
    )

# %% [markdown]
# The strict run pairs train frames 39-48 with test frames 9-18. Those frames do correspond: test frame 9 is train
# frame 39, and the offset of 30 is the same one the relaxed run reports. The strict run has found a correct stretch.
# It has found half of one. The whole test clip comes from train frames 30-49, so acting on the strict result alone
# drops 10 frames and leaves the other 10 leaked frames in the test split.
#
# ```{note}
# Start at `hash_radius=6` for video. The perceptual hash is 64 bits wide. DataEval reads a distance of 1-5 bits as
# highly similar and 6-10 bits as possibly similar, and published video hashing work puts its threshold near 10% of
# the hash length, which is about 6 bits here. The default stays at `0` so that the parameter means the same thing on
# every dataset, rather than changing behavior depending on what it was handed.
#
# `Duplicates` logs a warning when it runs on video with `hash_radius=0`. To see it, configure the standard library
# `logging` module as described in [How to configure logging](./h2_configure_logging.py):
#
# ```python
# import logging
# logging.basicConfig(level=logging.WARNING)
# ```
# ```

# %% [markdown]
# ## Triaging the corpus
#
# Start with the per-sequence summary. It gives one row per sequence and tells you which sequences are worth
# investigating.

# %%
summary = relaxed.aggregate_by_sequence()
display(summary)

# %% [markdown]
# Two of these columns measure duplication:
#
# - `redundant_fraction` measures self-redundancy: the fraction of frames that carry nothing new over the frames
#   immediately before them. Sequence 2 scores highest, from its 20-frame camera dwell.
# - `duplicate_frames` measures cross-sequence duplication: frames of this sequence that also appear in a
#   different sequence. Sequences 0 and 1 report all 60 of their frames, since each is a copy of the other. Sequence 3
#   reports the 20 frames it reuses from sequence 0, and sequence 2 reports none.
#
# The other two are counts rather than identifiers:
#
# - `shared_with` is how many other sequences this one shares content with, not which ones. Sequence 0 reads `2`,
#   because sequence 1 copies it whole and sequence 3 reuses a stretch of it. Read the sequence-level rows below to
#   find out which sequences those are.
# - `group_count` is how many duplicate groups touch this sequence, counting the redundant runs inside it. It
#   tells you how fragmented the findings are, not how much footage is duplicated: one long dwell is a single group,
#   while the same number of frames spread over ten short runs is ten.
#
# `duplicate_frames` counts only frames matched in another sequence. Consecutive frames of almost any video
# resemble one another within a usable `hash_radius`, so counting that resemblance here would make a corpus of
# unrelated videos read as fully duplicated. It is self-redundancy, and `redundant_fraction` already reports it.

# %% [markdown]
# ## Identifying sequence-level duplicates
#
# The summary tells you that a sequence shares content. It does not tell you which sequence it shares with, how much,
# or where. The sequence-level rows answer that.

# %%
display(
    relaxed.sequences.data().select(
        "dup_type", "item_indices", "span_start", "span_end", "containment", "mean_distance"
    )
)

# %% [markdown]
# The three rows cover two different relationships:
#
# - Sequences 0 and 1 overlap from frame 0 to frame 59, and each one fully contains the other (`containment` is
#   `[1.0, 1.0]`) at a mean distance of about 1 bit per frame. That is a re-encode of the same footage. Keep one of
#   the two and drop the other.
# - Sequences 0 and 3 share a 20-frame stretch: frames 5-24 of sequence 0 are frames 15-34 of sequence 3. Neither
#   sequence contains the other, and `containment` sits near `0.4` on both sides. This is the case where part of one
#   sequence duplicates part of another, and it is reported the same way inside one dataset as it is across two.
# - The third row is that same stretch found against sequence 1, which is itself a copy of sequence 0. A reused clip
#   is reported against every copy of its source.
#
# `containment` is the share of each sequence's frames that matched, counted over every match rather than over the
# reported span alone. It can sit slightly above the span length divided by the sequence length when frames outside
# the span match too at an offset that does not extend the segment.
#
# The `dup_type` column names the kind of relationship:
#
# | `dup_type` | Description |
# | --- | --- |
# | `exact` | The two sequences hold identical frames in identical order. |
# | `segment` | The two sequences run together over a continuous stretch at a fixed frame offset. |
# | `aligned` | The two sequences run together but not at the same rate, which is a speed edit. See below. |
# | `redundant` | A continuous run inside one sequence that carries nothing new over the frames before it. |

# %% [markdown]
# ## Detecting train-test leakage
#
# Footage shared between the training and test splits inflates your measured performance. Pass both splits to
# `evaluate`, and the `dataset_indices` column records which split each side of a match came from.

# %%
leakage = Duplicates(hash_radius=6, min_segment_frames=10).evaluate(train, test)
leaks = leakage.crossing.aggregate_by_pair("sequence")
display(leaks)

# %% [markdown]
# The `.crossing` attribute keeps only the relationships that cross the dataset boundary. `aggregate_by_pair` gives
# one row per matching pair and puts each dataset's containment in its own column.
#
# Read the two containment columns together, because duplication between a long video and a short clip is asymmetric.
# A `containment_a` of `0.4` against a `containment_b` of `1.0` says the shared stretch is 40% of the training
# sequence and 100% of the test clip. A single symmetric similarity score cannot say that, and a transitive duplicate
# group loses it entirely, which is how a fully leaked test sequence goes unnoticed.
#
# The second row is the same clip found in the transcoded training sequence.
#
# For the frame numbers themselves, read the sequence-level rows. Spans are reported in source-video coordinates:

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
# Spans are reported as source-video frame numbers, matching the coordinates in `unit_indices`. Seeking to a reported
# frame in a video player lands on the duplicated footage, whatever frame sampling the evaluation used.
# ```

# %% [markdown]
# ## Identifying redundant video segments
#
# `redundant_fraction` tells you how much of a sequence repeats itself. To act on it you need the intervals. Filter
# for the redundant rows and sort them by length.

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
# The longest run is the 20 frames of sequence 2, at a `mean_distance` of `0.0`. That is the camera dwell, where the
# same frame repeats exactly. The shorter runs sit at 2 to 4 bits, which is slow footage rather than a held camera.
# Treat the two differently: a dwell costs you storage and annotation for frames that add nothing, while slow footage
# is still moving and still carries information.
#
# This is why `aggregate_by_sequence` reports `longest_run` next to `redundant_fraction`. Two sequences can score the
# same fraction and be nothing alike, one holding still once and the other moving slowly throughout:

# %%
display(summary.select("sequence", "redundant_fraction", "longest_run"))

# %% [markdown]
# A static run of *k* frames can be cut to a single frame without losing content, and `redundant_frames` counts what
# that would save. Check before you cut: dwell time can be signal, and a tracker trained only on moving objects can
# fail on stationary ones.

# %% [markdown]
# ## Detecting repackaged or resampled footage
#
# Segment matching looks for a constant frame offset between two sequences. Re-export or resample a video at another
# frame rate and that offset drifts, so the shared footage arrives as fragments too short to report and the default
# settings find nothing.

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
# `verify_alignment` turns on dynamic time warping, which matches two sequences that run together without keeping
# step. The `aligned` row pairs the source with the slowed export at a mean distance of a little over one bit per
# frame.
#
# The value you give `verify_alignment` is the largest mean Hamming distance per aligned frame you will accept. Start
# at `8` for perceptual hashes. The check is off by default because warping is quadratic in the two sequence lengths,
# against a near-linear segment search. DataEval runs it only on the pairs the segment search could not explain.

# %% [markdown]
# ## Identifying duplicated annotations
#
# A track is one object followed across consecutive frames. Setting `levels="track"` hashes the cropped object images
# instead of whole frames and looks for the same relationships between tracks that it looks for between sequences.
# Two tracks match when their crops match over a stretch of frames.
#
# `levels` names the granularities to evaluate, and any level you leave out is not computed. This run skips
# frame-level work entirely and reports track relationships only.

# %%
tracked = Duplicates(hash_radius=6, min_track_frames=10).evaluate(train, levels="track")
display(
    tracked.tracks.data().select("dup_type", "item_indices", "track_indices", "span_start", "span_end", "containment")
)

# %% [markdown]
# Every row pairs sequence 0 with sequence 1, because the copy carries the same annotations as its source. The
# `track_indices` column gives the identifiers as they appear in your annotations rather than an internal
# renumbering, so you can find the flagged tracks in the source data.
#
# What a track match means depends on where the two tracks sit:
#
# - In two different sequences, as here, the underlying footage is shared. Either a clip was reused, or objects
#   were copy-pasted between clips as augmentation. The annotations are not wrong, so do not re-annotate them. Remove
#   the reused clip, or, if the objects were pasted in deliberately, record that and leave the tracks alone.
# - In one sequence, over the same frames, two identifiers are following one object. Merge the identifiers in the
#   annotations.
#
# Track matching compares appearance, so it finds tracks that show the same object over the same frames. It does not
# find a track split along time, where one object is identifier 4 for thirty frames and identifier 9 afterwards: the
# two halves show different frames of the object and hash differently. Finding those needs temporal reasoning over
# the annotations rather than appearance matching.
#
# `min_track_frames` sets the shortest matching stretch to report and defaults to `5`, separately from
# `min_segment_frames`. Thirty frames is a reasonable bar for a shared video clip. A single object tracked for
# thirty frames is already a long track.
#
# ```{note}
# The `levels` parameter and the `per_image`/`per_target` parameters are two ways of asking for the same thing.
# `per_target=True` is equivalent to asking for track-level relationships, since it computes the crop-level hashes
# they read. Do not pass both: `Duplicates` raises a configuration error rather than reconciling them for you.
# ```

# %% [markdown]
# ## Mapping results back to the dataset
#
# Every duplicate is reported in source-video coordinates, as a sequence index and frame numbers. To view the
# duplicated frames, pull out a sequence, or build a split without them, pass those coordinates to
# :class:`~dataeval.data.FrameIndices`, which `frame_sample` accepts.
#
# To pull the leaked stretch, frames 30-49 of training sequence 0:

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
# The same selector gives you the complement, the footage you want to keep. DataEval reports where the sequences
# overlap; which side to cut is your decision.

# %%
n_frames = len(train[sequence][1].frame_tracks)
keep = [frame for frame in range(n_frames) if not first <= frame <= last]
without_leak = SequenceFrames(train, FrameIndices({sequence: keep}))
print(f"sequence {sequence} without the shared stretch: {len(without_leak)} of {n_frames} frames")

# %% [markdown]
# Group members work the same way. `result.frames.exact` returns a list of `(sequence, frame)` pairs, so collect them
# by sequence and hand the mapping to `FrameIndices`:
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
# `FrameIndices` is lazy. Building a view decodes nothing until you read from it, and `len()` stays cheap. Frames
# come back in ascending order per sequence whatever order you list them in, because video streams are read forwards
# without rewinding.
# ```

# %% [markdown]
# ## Acting on the results
#
# The relationships map onto a handful of curation decisions:
#
# | What you found | What to do |
# | --- | --- |
# | `containment` near `[1.0, 1.0]` between two sequences | The same footage stored twice. Keep one, preferring the copy with better annotations. |
# | Lopsided `containment` across two splits | Leakage. Cut the shared frames from the test split, or split by sequence instead. |
# | A `segment` shared by two sequences in one split | A reused clip. Keep one copy of the stretch, or keep both and make sure the split keeps them together. |
# | High `redundant_fraction` | Thin the sequence with `frame_sample` rather than deleting frames, after checking that the dwell is not itself signal. |
# | An `aligned` relationship | Repackaged or resampled footage. Treat the aligned stretch as a duplicate of the source. |
# | A track match across two sequences | Shared footage, not a labeling error. Remove the reused clip, or record it if the objects were pasted in as augmentation. |
# | A track match inside one sequence | Two identifiers on one object. Merge them in the annotations. |
#
# Split by sequence. `SequenceFrames` exposes `sequence` as a unit-level metadata attribute, so
# `split_dataset(..., split_on=["sequence"])` keeps every frame of a video on one side of the split and stops you
# reintroducing the leakage you just found.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# hash_radius matters: the strict default under-reports the relation rather than missing it
strict_pair = strict.sequences.data().filter(pl.col("item_indices") == [0, 1]).row(0, named=True)
assert strict_pair["containment"][0] < 0.6

# three relations: the whole-sequence copy, and the reused clip against both copies of its source
assert relaxed.sequences.data().shape[0] == 3
copied = relaxed.sequences.data().filter(pl.col("item_indices") == [0, 1]).row(0, named=True)
assert copied["containment"] == [1.0, 1.0]
reused = relaxed.sequences.data().filter(pl.col("item_indices") == [0, 3]).row(0, named=True)
assert (reused["span_start"], reused["span_end"]) == ([5, 15], [24, 34])
assert max(reused["containment"]) < 0.5

# the strict run finds a correct stretch of the leak, not a wrong one, and finds half of it
assert (extent[0]["span_start"][0], extent[0]["span_end"][0]) == (39, 48)
assert (extent[0]["span_start"][1], extent[0]["span_end"][1]) == (9, 18)
assert extent[6]["containment"][1] == 1.0
assert extent[0]["containment"][1] < 0.7

# self-repetition is not cross-sequence duplication
assert summary["duplicate_frames"].to_list() == [60, 60, 0, 20]
assert summary["shared_with"].to_list() == [2, 2, 0, 2]
assert summary["redundant_fraction"].to_list()[2] == max(summary["redundant_fraction"].to_list())

# leakage: the test clip is entirely drawn from training footage
assert leaks.shape[0] >= 1
first_leak = leaks.row(0, named=True)
assert (first_leak["dataset_a"], first_leak["dataset_b"]) == (0, 1)
assert first_leak["containment_b"] == 1.0
assert first_leak["containment_a"] < 0.5
spans = leakage.sequences.data().filter(pl.col("dataset_indices").list.n_unique() > 1).row(0, named=True)
assert (spans["span_start"][0], spans["span_end"][0]) == (30, 49)
assert (spans["span_start"][1], spans["span_end"][1]) == (0, 19)

# the dwell is the longest redundant run, and it repeats exactly
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

# the tracks came with the copied sequence
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
