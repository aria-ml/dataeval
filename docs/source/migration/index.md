# Migration

Version-to-version upgrade guides. Each one covers what changed for existing code, what to
change now, and what disappears in the following release.

DataEval retires a deprecated API one minor release after deprecating it, so a guide's
"removed in" notes are the deadline for migrating off it.

:::{toctree}
:hidden:

v1.3
v1.2
v1.1
:::

:::{list-table}
:widths: 20 80
:header-rows: 0

- - [](v1.3.md)
  - v1.2 to v1.3: experimental blur statistics for `image_stats` (`ImageStats.BLUR`) and `video_stats`
    (`VideoStats.BLUR`), and `Resize`, `SelectChannels("gray")` and `TorchvisionTransform` declare the image ones as
    invalidated. Spatial statistics (`ImageStats.SPATIAL`) compare a box with its neighbors. A saturation
    statistic (`ImageStats.PIXEL_SATURATION`, in `PIXEL`) reports the share of values at a clipping endpoint, so the default
    image statistics gain a column, and `DetectionCrops` declares it invalidated too. A dataset can declare each image's
    `value_range` in its metadata. `Duplicates`
    searches videos with long static stretches for shared segments with experimental `hash_neighbors` arguments
    `partition` and `k`, and a `segment_neighbors` option. An experimental `frame_stats` measures the time between frames
    and exact repeats, and `track_stats` returns two more columns, the longest gap and the net heading of a
    track.
- - [](v1.2.md)
  - v1.1 to v1.2: the removal of everything v1.1 deprecated (metadata level shims, the
    `selection` module, `per_channel` and the renamed preprocessing helpers, the
    `is_discrete` protocol fallback, `discrete_features` and
    `ClassifierUncertaintyExtractor`), and `SourceIndex` becomes an address:
    `SourceIndex.target` is renamed `key` and its third field becomes `level`.
- - [](v1.1.md)
  - v1.0 to v1.1: metadata levels, `View` and operations, image statistic scales, and
    chance-corrected `Balance`. Everything deprecated here is removed in v1.2.0.
:::
