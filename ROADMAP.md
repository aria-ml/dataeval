# DataEval Roadmap

**Last updated:** August 2026
**Horizon:** through February 2027

This roadmap captures the long-term vision and rough quarter-level timing for DataEval
through early 2027 (roughly one minor release). Detailed schedules live in PI planning;
this document is intentionally coarser so it stays meaningful longer.

---

## Vision

DataEval is the evaluation library for image and video datasets used in operational ML
systems. Three focuses drive the work through this horizon:

1. **Full-motion video (FMV) expansion.** FMV-native metrics with no still-image analogue
   are added, with a focus on multiobject tracking data analysis and validation.
2. **Ontology and label validation as first-class capabilities.** Label taxonomies are a
   core input to the library, with validation, alignment, and taxonomy-aware analyses
   available to every downstream evaluator.
3. **Operational adoption.** Demos and reference workflows take the library from
   research-grade to production-usable on operational datasets.

---

## Releases

| Release | Date       | Theme                                                      |
|---------|------------|------------------------------------------------------------|
| v1.0    | Mar 2026 ✓ | Quality, performance, bias, and shifts modules; API freeze |
| v1.1    | Aug 2026 ✓ | Scope module; ontology stack; object-tracking foundation   |
| v1.2    | Q1 2027    | FMV foundation and first video evaluators; ontology depth  |

Minor releases have run about five months apart - v1.0 in March 2026, v1.1 in August
2026. v1.2 is planned against that observed cadence.

---

## Shipped in v1.1 (August 2026)

**Scope module.** `Coverage`, `Prioritize`, and `Representation` evaluators, with the
supporting core functions for adaptive and naive coverage, completeness, and label
coverage.

**Ontology stack.** An `Ontology` type built from RDF or from a plain hierarchy, with
taxonomy queries (ancestors, descendants, siblings, subtrees, lowest common ancestor) and
`label_collisions` for detecting taxonomies whose surface forms conflict. Alongside it, a
set of core label functions: `label_alignment`, `label_coverage`, `label_errors`,
`label_parity`, `label_reconciliation`, and `label_stats`.

**Object-tracking foundation.** Track types, track-aware dataset views, per-track
statistics, and tracking-aware metadata structurers. This is the substrate the video work
below builds on - tracks are the first data model in the library whose identity spans
frames.

**Metadata and bias corrections.** Metadata levels rework, and chance correction
throughout `Balance` and `mutual_info` so that a finely binned factor no longer reports a
correlation with everything.

**Deferred out of v1.1.** Every video-native capability originally planned alongside the
tracking work moves to v1.2.

---

## v1.2 - target Q1 2027

This work lies along two tracks; The FMV track and the ontology and operational track.

### Track 1 - FMV foundation and first evaluators

Foundation:

- Video dataset classes, extending the track-aware data model shipped in v1.1.
- Key-frame selection, enabling the use of statistical tools on video frames.
- Ego-motion removal, facilitating unsupervised analysis and label error detection.
- Video-aware splitting, ensuring no two clips from the same source video land on
  opposite sides of a split.
- FMV statistics: motion, quality, and aggregated frame statistics.
- Formalized Aggregators, to ensure stats/metadata at different levels can be passed
  to bias detectors (and other tools) smoothly.

Video-specific evaluators that build on this foundation:

- **Quality on video** - duplicate and near-duplicate detection.
- **MOT labeling error detection** - spatially and temporally misaligned boxes, common
  track ID errors.

### Track 2 - Ontology depth and operational hardening

- Hierarchical-taxonomy support in the bias and balance evaluators, so a taxonomy's
  structure informs the grouping rather than only its leaf labels.
- Ontology drift detection across dataset versions.
- Operational demos: deploy and validate DataEval against operational data, with dataset
  exploration and curation ergonomics driven by what those workflows expose.
- Import and startup cost. `import dataeval` is dominated by eagerly importing torch and
  scikit-learn through `dataeval.core`; making the core module surface lazy is the first
  step and was deliberately deferred out of v1.1 as too broad for a release eve.

**Scope note.** Both tracks in one release is more than v1.1 carried. The advanced video
evaluators will likely move to the following release.

---

## Direction beyond v1.2

- **Full video validation and documentation.** Evaluator modules - shifts, scope, and
  performance - validated on video across classification, object detection, and object
  tracking, and proper usage documented.
- **More FMV-native metrics.** Measures with no direct still-image analogue, such as
  temporal drift within a video sequence, and additional stats for use in evaluators that
  analyze metadata.
- **Support for exotic video formats.** - Streaming (long video) datasets, multi-camera
  and multi-view datasets, real-time evaluation.

A v2.0 marking full video support across every evaluator module remains the destination.

---

## Success criteria for this horizon

- Video datasets are a first-class input: loadable, splittable, and describable with the
  same statistics still-image datasets already get.
- At least one evaluator module runs end to end on video and is validated on a benchmark
  dataset.
- Taxonomy structure, not just leaf labels, reaches the bias and balance evaluators.
- The library is exercised against operational data in relevant demos.
