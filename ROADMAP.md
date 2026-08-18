# DataEval Roadmap

**Last updated:** August 2026
**Horizon:** through February 2027

This roadmap covers the next six months, which at the cadence the project has actually
held is about one minor release. Work beyond that horizon is listed as direction only and
carries no dates, because dates that far out have not survived contact with the schedule.
Detailed planning lives in PI planning; this document is intentionally coarser so it stays
true longer.

---

## Vision

DataEval is the evaluation library for image and video datasets used in operational ML
systems. Three shifts drive the work through this horizon:

1. **Full-motion video (FMV) parity.** Every evaluator that exists for still images
   becomes available for video, with FMV-native metrics added where they have no
   still-image analogue.
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

Minor releases have run about five months apart — v1.0 in March 2026, v1.1 in August
2026. v1.2 is planned against that observed cadence rather than against a shorter target.

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
below builds on — tracks are the first data model in the library whose identity spans
frames.

**Metadata and bias corrections.** Metadata levels rework, and chance correction
throughout `Balance` and `mutual_info` so that a finely binned factor no longer reports a
correlation with everything.

**Deferred out of v1.1.** Every video-native capability originally planned alongside the
tracking work — dataset classes, key-frame and key-clip selection, framerate
normalization, video-aware splitting, and FMV statistics. All of it moves to v1.2.

---

## v1.2 — target Q1 2027

Two tracks. The FMV track is sequential: the foundation has to land before any evaluator
can run on video. The ontology and operational track is independent and runs alongside it.

### Track 1 — FMV foundation and first evaluators

Foundation, in dependency order:

- Video dataset classes, extending the track-aware data model shipped in v1.1.
- Key-frame and key-clip selection.
- Clipping and framerate normalization, so datasets collected at different rates become
  comparable.
- Video-aware splitting, ensuring no two clips from the same source video land on
  opposite sides of a split.
- Initial FMV statistics: time, motion, quality, and aggregated frame statistics.

First evaluators on video, once the foundation is stable:

- **Quality on video** — duplicate and near-duplicate detection, outlier detection, and
  core-set selection over video embeddings and motion signatures.
- **One bias evaluator with video-aware groupings**, to prove the grouping path end to
  end before committing every module to it.

### Track 2 — Ontology depth and operational hardening

- Hierarchical-taxonomy support in the bias and balance evaluators, so a taxonomy's
  structure informs the grouping rather than only its leaf labels.
- Ontology drift detection across dataset versions.
- Operational demos: deploy and validate DataEval against operational data, with dataset
  exploration and curation ergonomics driven by what those workflows expose.
- Import and startup cost. `import dataeval` is dominated by eagerly importing torch and
  scikit-learn through `dataeval.core`; making the core module surface lazy is the first
  step and was deliberately deferred out of v1.1 as too broad for a release eve.

**Scope note.** Both tracks in one release is more than v1.1 carried. If the FMV
foundation slips, Track 2 ships as v1.2 on its own and the video evaluators move to the
release after it.

---

## Beyond v1.2 — direction, no dates

- **Full video parity.** The remaining evaluator modules — shifts, scope, and performance
  research — validated on video across classification, object detection, and object
  tracking.
- **FMV-native metrics.** Measures with no direct still-image analogue: temporal drift
  within a clip and across collection time, scene complexity from entity counts and scene
  transitions, action diversity across action classes and transitions, camera motion
  characterization, temporal consistency of labels and embeddings, and occlusion and
  visibility profiles.
- **Video-specific evaluators.** Near-duplicate detection across videos, clips, and
  frames; a leakage detector for video datasets integrated into the bias module.
- **Scale.** Long-video support with streaming embeddings, multi-camera and multi-view
  datasets, real-time evaluation hooks.
- **Synthetic and augmented video.** Detection and characterization of synthetic content
  in operational datasets.

A v2.0 marking full video support across every evaluator module remains the destination.
It is not given a date here: at the observed cadence it is more than one release away, and
the last two roadmaps placed it a year earlier than the schedule supported.

---

## Success criteria for this horizon

- Video datasets are a first-class input: loadable, splittable, and describable with the
  same statistics still-image datasets already get.
- At least one evaluator module runs end to end on video and is validated on a benchmark
  dataset.
- Taxonomy structure, not just leaf labels, reaches the bias and balance evaluators.
- The library is exercised against operational data, with the resulting ergonomic gaps
  captured as tracked work rather than as folklore.
