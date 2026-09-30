# DataEval Roadmap

**Last updated:** September 2026
**Horizon:** v1.3 (Q1 2027), with a longer-term vision beyond it

This roadmap captures the long-term technical vision and release plan for the core
DataEval Python library. Detailed schedules live in PI planning; this document is
intentionally coarser so it stays meaningful longer. It is a living document: when a
release's scope moves, this page moves with it and says where the work went.

---

## Vision

DataEval is the core evaluation library for datasets used in operational ML systems,
built on high-fidelity, statistically rigorous metrics for dataset and model evaluation.
It covers still imagery today; video is the near-term expansion, and multi-sensor and
multi-modal data follow it.

Five technical pillars drive the long-term vision:

1. **Advanced FMV and intrinsic video metadata.** Moving from still-image analysis to
   FMV-native metrics, with a focus on multiobject and multiple-hypothesis tracking (MHT),
   time-series data quality, temporal leakage detection, and container- and codec-level
   intrinsic video metadata (such as bitrate variation, frame types, and motion vectors)
   for evaluating raw ingestion quality.
2. **Ontology and label validation as first-class capabilities.** Taxonomies as first-class
   citizens of the library: ontology compliance checking, semantic alignment, completeness
   validation, and taxonomy-aware analyses available to every downstream evaluator.
3. **Simulated, synthetic, and augmented data metrics.** Evaluation paradigms for generative
   models and synthetic datasets, including metrics that prioritize data augmentation and
   predict downstream model improvement from synthetic samples.
4. **Multi-modal data support.** Extending beyond computer vision to text, audio, tabular,
   and joint multi-modal datasets, with standardized representations, bias detection, and
   cross-modal alignment metrics.
5. **Enterprise scalability and large-scale integration.** Core performance and scalability
   work, so the library runs efficiently against large-scale datasets and cloud/lakehouse
   platforms (e.g., Databricks).

Two principles run through all five:

- **Metadata is the common layer.** Factors live at a `sequence`, `unit`, `track`, or
  `instance` level, roll up between levels by declared aggregations, and resolve back to
  the source item they came from. A new statistic or a new modality reaches every
  evaluator by producing factors, not by adding evaluator-specific plumbing.
- **Statistics follow the data's structure.** When rows are not independent (300 frames of
  one clip, or one collect filmed at 60 fps beside another at 10 fps), significance and
  chance corrections count the independent entities, not the rows.

---

## Releases

| Release | Date       | Theme                                                                                          |
|---------|------------|------------------------------------------------------------------------------------------------|
| v1.0    | Mar 2026 ✓ | Quality, performance, bias, and shift modules; API freeze                                      |
| v1.1    | Aug 2026 ✓ | Scope and ontology stacks; object-tracking foundation; MAITE protocol adoption                 |
| v1.2    | Q4 2026    | Video-ready metadata: level aggregation, structure-aware statistics, metadata-aware quality    |
| v1.3    | Q1 2027    | Evaluators on video; intrinsic video quality and codec metadata; annotation diagnostics        |
| Future  | Long-term  | Ontology depth; weighted and output-driven evaluation; multi-modal; synthetic data; scale      |

Releases are cut when a coherent capability is complete and its public API has settled,
not on a fixed interval. v1.1 followed v1.0 by five months; v1.2 follows v1.1 by about
three, because the metadata work reached a natural boundary ahead of the video evaluators
that build on it.

---

## Shipped in v1.1 (August 2026)

**Scope module.** The `Coverage` and `Representation` evaluators join `Prioritize`, which
shipped in v1.0, with the supporting core functions for adaptive and naive coverage,
completeness, and label coverage.

**Ontology stack.** An `Ontology` type built from RDF/OWL or from a plain hierarchy, with
taxonomy queries (ancestors, descendants, siblings, subtrees, lowest common ancestor) and
`label_collisions` for detecting taxonomies whose surface forms conflict. Alongside it, a
set of core label functions — `label_alignment`, `label_coverage`, `label_errors`,
`label_parity`, `label_reconciliation`, and `label_stats` — plus `ontology_validation`,
which reports the structural and naming defects of an ontology artifact itself.

**Object-tracking foundation.** Track types, track-aware dataset views, per-track
statistics, and tracking-aware metadata structurers. Tracks are the first data model in
the library whose identity spans frames.

**Metadata restructure.** Metadata levels reworked into an explicit
`sequence`/`unit`/`track`/`instance` schema that is no longer vision-specific. A
`sequence` is a video: one dataset item holding an ordered run of frames. This is the
substrate for the video work that follows and for the multi-modal work beyond it.

**MAITE protocol adoption.** MAITE is now a direct dependency rather than a set of
internal mirrors, with multi-object-tracking protocol support (`maite>=0.9.4`) and
registered `maite.tasks` and `maite.protocols` model entry points. Interoperability with
MAITE-compliant datasets and models predates v1.1; what is new is that DataEval consumes
the protocols directly.

**Bias corrections.** Chance correction throughout `Balance` and `mutual_info`, so that a
finely binned factor no longer reports a correlation with everything.

---

## v1.2 — target Q4 2026

**Theme: video-ready metadata.** v1.1 introduced the metadata levels; v1.2 makes them
carry analysis. Factors roll up from frame to track to sequence, the bias statistics stop
reading 300 frames of one clip as 300 independent samples, and the quality evaluators
read metadata at every level. Most of this is already merged. The release waits only on
the items that would otherwise change a public API after it ships.

### Metadata across levels

- **Aggregation.** `Metadata.aggregate` rolls factors from a finer level to a coarser one,
  including temporal reductions (variance, trend, and run length), and statistics
  producers declare how their own factors roll up, so frame statistics reach track and
  sequence level without hand-written reductions.
- **Corrections and repair.** `Metadata` accepts mixed inputs and applies declared
  corrections; `Metadata.repair` returns a corrected copy, and `Metadata.unusable_rows`
  names the items and rows behind a factor that could not be used.
- **Metadata as an analysis input.** `Metadata.classed_by` makes any factor the class axis
  for the bias evaluators, and `Metadata` feeds the drift detectors directly as a feature
  extractor.
- **A universal address.** `SourceIndex` identifies a row at any level (sequence, frame,
  track, or instance), and `SourceLocator` retrieves the source item behind it.

### Statistics that follow the data's structure

- `mutual_info`, `Balance`, and `Parity` count distinct entities rather than rows when they
  correct for chance and judge significance, so replication no longer reads as
  correlation.
- `Parity` computes Pearson's chi-square and judges sufficiency by Cochran's rule.

### Video data handling

- **Frame selection.** `SequenceFrames` and its selectors (`Stride`, `EvenlySpaced`,
  `FrameRate`, `Window`, `Cuts`, `Redundancy`, `Representative`) turn a video into the
  frames the statistical tools analyze, and record how much of the video each selected
  frame stands for.
- **Segments and stitching.** `VideoSegments`, `VideoStitch`, and `SegmentPlanner` cut long
  videos into clips and reassemble them.
- **Video statistics and embeddings.** `ego_stats` measures per-frame camera motion,
  `track_stats` runs directly on a tracking dataset, and `VideoTorchExtractor` embeds
  clips.
- **Up-front validation.** `validate_dataset` rejects a MOT dataset whose per-frame
  detection arrays disagree, before any frame is decoded.

### Quality evaluators on metadata and video

- **Duplicates** finds a collision in any projection: content, metadata factors, or
  annotations. Annotation fingerprints report a collect carrying two disagreeing
  annotations, and augmented copies that share an annotation but differ in pixels. On
  tracking datasets it matches whole sequences and segments as well as frames.
- **Outliers** thresholds metadata factors as well as image statistics, each at its own
  level (frame, track, or sequence), and box statistics report geometry relative to the
  image.

### Also in v1.2

- Statistics over band groups, for multispectral channels.
- `ChunkedDrift` accepts any chunker the protocol describes and handles a short final
  chunk explicitly.
- Bag-of-visual-words extraction runs on threads and scales with `set_max_processes`.
- **Breaking:** Python 3.11 or later is required, and the functionality deprecated in v1.1
  is removed.

### Remaining before release

- Metadata in `Outliers` (in review), and restraint when a level has too few rows to
  threshold.
- A settled frame-selection API: one entry point where `Duplicates` currently takes both a
  `frame_sample` and a `FrameSelector`, and defined semantics for chaining selectors.
- The announced change to the image `hash_radius` default in `Duplicates`, and
  rotation- and flip-invariant (D4) hashes reaching sequence, segment, and track matching.

### Moved out of v1.2

The August revision of this roadmap planned more for v1.2 than the release above holds.

- **To v1.3:** container and codec metadata, video quality statistics, video-aware
  splitting, ego-motion removal, MOT labeling error detection, and import and startup
  cost.
- **Unscheduled:** the ontology track (hierarchical taxonomies in the bias evaluators,
  ontology drift, compliance and alignment evaluators, and ontology coverage over metadata
  hierarchies) and generalizing the estimators to any classifier output. Neither is in
  current planning; both are described under [Direction beyond v1.3](#direction-beyond-v13).

---

## v1.3 — target Q1 2027

v1.3 runs the evaluators on video and measures what is intrinsic to it. The work lies
along four threads.

### Evaluators on video

- **Duplicates profiles and throughput.** Named profiles for the three questions users
  bring (leakage, collect-level deduplication, and redundancy within a sequence) in place
  of four interacting radius and tolerance parameters, and a hash search that scales past
  all pairs.
- **Video leakage and splitting.** Detect clips of one source video, or near-duplicate
  footage, on both sides of a split, and carry sequence identity through splitting so
  that it cannot happen.
- **Outliers in context.** Judge a value within its category or against its neighbors in
  the sequence (a timestamp that jumps three hours for one frame), and flag combinations
  that are ordinary column by column but improbable together.
- **Bias and coverage on video.** `Balance` and `Diversity` over track- and
  sequence-level factors; `Coverage` and `Representation` over clip embeddings.
- **Drift on video.** Drift over clip embeddings and over aggregated video statistics,
  starting from a decision on the unit of drift: frame, key frame, clip, or sequence.
- **Selection criteria.** Budget-constrained frame selection, and pixel-domain criteria
  such as inter-frame difference and a sharpness gate.
- **Guides** for applying each tool to video data.

### Intrinsic video metadata

- **Descriptive statistics:** resolution and a sequence-level statistic family.
- **Motion:** camera motion magnitude, building on `ego_stats`.
- **Quality:** compression (blockiness, GOP length and variability, interlacing, frame
  independence, quantization) and artifacts (noise, flicker, flashing, ringing, chromatic
  aberration, ghost contours), each with guidance on reading it.
- **Container and codec metadata:** codec, GOP structure, frame types, and quantization
  read from the encoded stream, so capture and transmission anomalies surface without
  full decoding. Datasets that yield only decoded frames do not carry the encoded stream;
  for those, the statistics that can be estimated from frames fall back to that path.
- **Fingerprinting, scene, and occlusion characteristics**, at investigation stage.

### Annotation diagnostics for video

Video annotations fail in ways still images cannot: auto-labelers deployed at the edge lag
the frames they label, boxes drift, tracks gap, and detections appear where nothing is.

- A single track-gap taxonomy separating occlusion, frame exit, missed label, and ghost,
  shared by the tools below so that they cannot reach contradictory verdicts on the same
  gap.
- Temporal and spatial alignment of annotations to frames.
- Missing and ghost detections, using ego-motion-removed video.
- Matching regions and points of interest to bounding boxes.
- Identity diagnostics (ID switches and fragmentation) from per-detection appearance
  embeddings.

**Scope note.** This thread starts from investigation. Evaluators ship as each method
proves out on the planted-defect evaluation corpus, and some will land after v1.3.

### Provenance and operational hardening

- Document the provenance of every feature: the method it implements, the reference it
  follows, and how it is validated. A gap analysis against that record, and the fixes it
  calls for.
- Import and startup cost. `import dataeval` is dominated by torch, imported eagerly by
  `dataeval.config`; the scikit-learn and scipy chain reached through `dataeval.core` is
  the next largest block. Making both lazy has been deferred twice as too broad for a
  release eve, and belongs early in a cycle.

---

## Direction beyond v1.3

The long-term focus is comprehensive, scalable, data-centric AI/ML evaluation across
advanced computer vision, temporal, generative, and multi-modal workflows. The first five
directions follow the pillars; the last two cut across them.

### 1. Advanced FMV and intrinsic video metadata

- **Advanced track analysis and MHT**: extend the tracking-aware tools to
  multiple-hypothesis tracking analyses and time-series alignment, generalizing the
  annotation diagnostics begun in v1.3.
- **Stream health**: evaluate raw video stream health, packet loss and corruption, and
  compression artifacts using motion vector statistics and container-level signaling.
- **Tracking drift**: detect tracker behavior degrading over time.

### 2. Ontology depth

Moved from v1.2 and not yet scheduled. The v1.1 ontology stack supplies the facts; this
work turns them into analyses.

- **Hierarchical taxonomies in bias and balance**, so a taxonomy's structure informs the
  grouping rather than only its leaf labels.
- **Ontology drift** across dataset versions.
- **Compliance and alignment evaluators** on top of `ontology_validation`,
  `label_alignment`, and `label_reconciliation`, turning their findings into policy
  verdicts against a specified reference ontology.
- **Ontology-based coverage over metadata hierarchies**, the remaining axis beyond
  `label_coverage` and `Representation`.

### 3. Simulated, augmented, and synthetic data metrics

- **Dataset augmentation guidance**: coverage- and prioritization-based evaluators that
  determine when, where, and how to augment real-world datasets with synthetic data.
- **Synthetic data quality metrics**: metrics that evaluate synthetic datasets by their
  predicted downstream model performance improvement.
- **Generative model evaluation**: metrics and tools for assessing and benchmarking
  generative models directly.

### 4. Multi-modal, text, audio, and tabular support

The v1.1 metadata restructure removed the vision-specific assumptions from the metadata
layer; the work below builds the modality-specific metrics on top of it.

- **Joint multi-modal alignment**: evaluation of alignment, semantic coherence,
  representation bias, and distribution shift in combined multi-modal datasets (e.g.,
  image-text, audio-video, speech-text).
- **Text modality support**: metrics for text coverage, vocabulary representation, semantic
  drift, topic representation, and text quality anomalies in NLP and LLM datasets.
- **Audio modality support**: tools for acoustic quality, signal-to-noise ratio, spectral
  coverage, clipping, and background noise in speech and audio-processing datasets.
- **Tabular modality support**: metrics for high-dimensional feature interaction, structural
  completeness, distribution shift, and representation balance across arbitrary tabular
  formats.

### 5. Large-scale integration and cloud scalability

- **Scalability**: design and optimize execution performance across all core evaluators to
  handle massive datasets.
- **Lakehouse connectors**: native interfaces from DataEval to large-data lakehouses and
  cloud data-platform APIs (e.g., Databricks).

### 6. Weighted and output-driven evaluation

- **Sample weights in the bias evaluators.** Frame selection, variable frame rates, collect
  structure, and resampling views all make rows stand for unequal amounts of the world.
  `SequenceFrames` already records how much of a video each frame represents; `Balance`,
  `Diversity`, `Parity`, and `mutual_info` do not yet read it. The gap predates video and
  applies to resampled image data as well.
- **Evaluation from model outputs alone.** A predictions table (scores and ground truth,
  with no raw data or runnable model) as a first-class input, with coverage, parity,
  uncertainty, and label-quality analyses applied through it. Every finding must read
  "your dataset has X", not "your model is Y".

### 7. Automated data remediation

Expand the image and video metrics and detectors to a broader range of distortions, and
provide algorithms that correct the data and label anomalies they detect, not only report
them. `Metadata.repair` in v1.2 is the first step on the metadata side.

---

## Long-term success criteria

- **Video and tracking are first-class.** Temporal, FMV-native, and MHT-native evaluators
  are validated on benchmark datasets.
- **Intrinsic video quality is measurable.** Container- and codec-level metadata is parsed
  and used to flag quality anomalies without full decoding.
- **Taxonomies inform the analysis.** Bias, coverage, and drift evaluators read an
  ontology's structure, not only its leaf labels.
- **Modalities beyond vision are supported.** General-purpose evaluators extend to text,
  audio, tabular, and joint multi-modal alignment checking.
- **Label and metadata deficiencies are remediable.** Core algorithms detect and correct
  them, not only report them, at pipeline scale.
- **Large-scale execution is practical.** High-performance execution against operational
  data, with native interfaces to cloud-lakehouse formats (e.g., Databricks).
- **Synthetic data is evaluable.** Standardized metrics for synthetic data quality and
  generative model assessment are implemented and validated.
