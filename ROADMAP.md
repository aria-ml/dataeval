# DataEval Roadmap

**Last updated:** August 2026
**Horizon:** v1.2 (early 2027), with a longer-term vision beyond it

This roadmap captures the long-term technical vision and release cadence for the core
DataEval Python library. Detailed schedules live in PI planning; this document is
intentionally coarser so it stays meaningful longer.

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

---

## Releases

| Release | Date       | Theme                                                                           |
|---------|------------|---------------------------------------------------------------------------------|
| v1.0    | Mar 2026 ✓ | Quality, performance, bias, and shift modules; API freeze                       |
| v1.1    | Aug 2026   | Scope and ontology stacks; object-tracking foundation; MAITE protocol adoption  |
| v1.2    | Q1 2027    | FMV foundation and key frames; intrinsic metadata; ontology depth and alignment |
| Future  | Long-term  | Multi-modal support (text, audio, tabular); advanced FMV; synthetic data; scale |

Minor releases run about five months apart (v1.0 in March 2026, v1.1 in August 2026).
v1.2 is planned against that observed cadence and program milestones.

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
statistics, and tracking-aware metadata structurers. This is the substrate the video work
below builds on — tracks are the first data model in the library whose identity spans
frames.

**Metadata restructure.** Metadata levels reworked into an explicit
`sequence`/`unit`/`track`/`instance` schema that is no longer vision-specific. A
`sequence` is a video: one dataset item holding an ordered run of frames. This is the
substrate for both the video work in v1.2 and the multi-modal work beyond it.

**MAITE protocol adoption.** MAITE is now a direct dependency rather than a set of
internal mirrors, with multi-object-tracking protocol support (`maite>=0.9.4`) and
registered `maite.tasks` and `maite.protocols` model entry points. Interoperability with
MAITE-compliant datasets and models predates v1.1; what is new is that DataEval consumes
the protocols directly.

**Bias corrections.** Chance correction throughout `Balance` and `mutual_info`, so that a
finely binned factor no longer reports a correlation with everything.

**Deferred out of v1.1.** Every video-native capability originally planned alongside the
tracking work moves to v1.2.

---

## v1.2 — target Q1 2027

This work lies along two tracks: the FMV track, and the ontology and operational track.

### Track 1 — FMV foundation, intrinsic metadata, and first evaluators

Foundation:

- Video dataset classes, extending the track-aware data model and the `sequence` metadata
  level shipped in v1.1.
- Key-frame selection, enabling the use of statistical tools on video frames.
- Ego-motion removal, facilitating unsupervised analysis and label error detection.
- Video-aware splitting. Grouped splitting already exists (`split_on` over metadata
  factors); what is missing is carrying sequence identity through automatically, so that
  no two clips from the same source video land on opposite sides of a split.
- FMV statistics: motion, quality, and aggregated frame statistics.
- **Intrinsic video metadata extraction**: parse container- and codec-level metadata (such
  as bitrate variation, frame types, GOP structure, and compression ratios) to evaluate
  video quality and capture or transmission anomalies without full decoding.
- Formalized aggregators, to ensure stats and metadata at different levels can be passed
  to bias detectors (and other tools) smoothly.

Video-specific evaluators that build on this foundation:

- **Quality on video** — duplicate and near-duplicate detection.
- **MOT labeling error detection** — spatially and temporally misaligned boxes, common
  track ID errors.

### Track 2 — Ontology depth and metric generalization

- Hierarchical-taxonomy support in the bias and balance evaluators, so a taxonomy's
  structure informs the grouping rather than only its leaf labels.
- Ontology drift detection across dataset versions.
- **Compliance and alignment evaluators**. The core algorithms shipped in v1.1
  (`ontology_validation`, `label_alignment`, `label_reconciliation`); what v1.2 adds is the
  evaluator layer on top, turning those facts into policy verdicts against a specified
  reference ontology.
- **Ontology-based coverage over metadata hierarchies**. Coverage and completeness across
  an ontology's classes shipped in v1.1 (`label_coverage` and `Representation`); the
  remaining axis is hierarchical metadata.
- **Generalization of metrics**: broaden algorithms and interfaces to a wider class of
  data — balance metrics on any labeled data, and performance-based estimation on any
  classifier output.
- Operational hardening and demos: validate DataEval against operational datasets, driving
  ergonomics and API improvements based on real-world ingestion.
- Import and startup cost. `import dataeval` is dominated by torch, imported eagerly by
  `dataeval.config` and accounting for more than half the total; the scikit-learn and scipy
  chain reached through `dataeval.core` is the next largest block. Making both surfaces
  lazy was deliberately deferred out of v1.1 as too broad for a release eve.

**Scope note.** Both tracks in one release is more than v1.1 carried. The advanced video
evaluators will likely move to the following release.

---

## Direction beyond v1.2

The long-term focus is comprehensive, scalable, data-centric AI/ML evaluation across
advanced computer vision, temporal, generative, and multi-modal workflows.

### 1. Advanced FMV and intrinsic video metadata

- **Advanced track analysis and MHT**: extend the tracking-aware tools to multiple-hypothesis
  tracking analyses and time-series alignment, generalizing the MOT error modeling delivered
  in v1.2.
- **Intrinsic quality modeling**: evaluate raw video stream health, packet loss and
  corruption, and compression artifacts using motion vector statistics, container-level
  signaling, and codec characteristics.
- **Temporal leakage and drift**: evaluators for temporal data quality, including detection
  of temporal leakage across splits in video sequences and of tracking drift over time.

### 2. Multi-modal, text, audio, and tabular support

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

### 3. Automated data remediation

- **Label and metadata correction**: expand the existing image and video metrics and
  detectors to a broader range of distortions, and provide algorithms that correct the
  data and label anomalies they detect, not only report them.
- **Generalization of estimators**: extend estimators to accept and analyze arbitrary model
  outputs, supporting automatic detection-to-correction pipelines.

### 4. Simulated, augmented, and synthetic data metrics

- **Dataset augmentation guidance**: coverage- and prioritization-based evaluators that
  determine when, where, and how to augment real-world datasets with synthetic data.
- **Synthetic data quality metrics**: metrics that evaluate synthetic datasets by their
  predicted downstream model performance improvement.
- **Generative model evaluation**: metrics and tools for assessing and benchmarking
  generative models directly.

### 5. Large-scale integration and cloud scalability

- **Scalability**: design and optimize execution performance across all core evaluators to
  handle massive datasets.
- **Lakehouse connectors**: native interfaces from DataEval to large-data lakehouses and
  cloud data-platform APIs (e.g., Databricks).

---

## Success criteria for this horizon

- **Video and tracking are first-class.** Temporal, FMV-native, and MHT-native evaluators
  are validated on benchmark datasets.
- **Intrinsic video quality is measurable.** Container- and codec-level metadata is parsed
  and used to flag quality anomalies without full decoding.
- **Modalities beyond vision are supported.** General-purpose evaluators extend to text,
  audio, tabular, and joint multi-modal alignment checking.
- **Label and metadata deficiencies are remediable.** Core algorithms detect and correct
  them, not only report them, at pipeline scale.
- **Large-scale execution is practical.** High-performance execution against operational
  data, with native interfaces to cloud-lakehouse formats (e.g., Databricks).
- **Synthetic data is evaluable.** Standardized metrics for synthetic data quality and
  generative model assessment are implemented and validated.
