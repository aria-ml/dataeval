# DataEval Roadmap

**Last updated:** May 2026

This roadmap captures the long-term vision and rough quarter-level timing for
DataEval through the end of 2026. Detailed schedules live in PI planning;
this document is intentionally coarser so it stays meaningful longer.

---

## Vision

DataEval is the evaluation library for image and video datasets used in
operational ML systems. Through the end of 2026 the focus is on four
strategic shifts:

1. **Full-motion video (FMV) parity.** Every evaluator that exists for still
   images becomes available for video, with FMV-native metrics added where
   they have no still-image analogue.
2. **Ontology and label validation as first-class capabilities.** Label
   taxonomies become a core input to the library, with validation, alignment,
   and taxonomy-aware analyses available to every downstream evaluator.
3. **Operational adoption.** Demos and reference workflows take the library
   from research-grade to production-usable on operational datasets.
4. **Program standards maturity.** SDP v1.2 and v1.3 compliance at ML1
   establishes the foundation for higher maturity levels in 2027.

---

## Library release targets

| Release | Target           | Theme                                                    |
|---------|------------------|----------------------------------------------------------|
| v1.0    | Feb 2026 ✓       | Quality, performance, bias, shifts modules; API freeze   |
| v1.1    | Q2 2026          | Scope module; ontology validation and alignment          |
| v1.2    | Q3 2026          | Initial FMV evaluators; coverage/completeness            |
| v2.0    | Q4 2026          | Full video support across every evaluator module         |

---

## Q2 2026 — current quarter

**Library v1.1 (May 2026).** Scope module (coverage and completeness) and
the first-class ontology stack: label-taxonomy validation (flat and
hierarchical, with reporting on unknown, ambiguous, and deprecated labels),
ontology alignment (normalizing labels across datasets that share a
taxonomy but use different surface forms), and integration points so
taxonomy-aware groupings flow into bias and balance evaluators.

**FMV foundation.** Video dataset classes, key-frame and key-clip selection,
clipping and framerate normalization, video-aware splitting. Initial FMV
statistics: time, motion, quality, and aggregated frame stats.

**SDP v1.2 @ ML1.** Repository structure, build/test/release procedures,
dependency and SBOM tracking, test coverage thresholds, static analysis and
vulnerability scanning in CI, release notes and version traceability.

---

## Q3 2026 — broaden FMV, advance SDP

**FMV evaluator coverage (library v1.2).** Each existing evaluator gets a
validated FMV path on benchmark datasets:

- Quality on video: outlier detection, duplicate detection, core set
  selection on video embeddings and motion signatures.
- Bias on video: parity, diversity, balance, and leakage detection with
  video-aware groupings.
- Shifts on video: OOD detection, divergence, and drift detection on video
  inputs (drift will likely require modification).
- Scope on video: coverage and completeness, including temporal coverage of
  concepts.
- Performance research: BER, UAP, and performance projection adapted to
  video tasks.

**FMV-native metrics.** New metrics with no direct still-image analogue:

- Temporal drift within a clip and across videos collected over time.
- Scene complexity from entity counts, motion vectors, and scene transitions.
- Action diversity across action classes and action-pair transitions.
- Camera motion characterization (pan/tilt/zoom, ego-motion).
- Temporal consistency of labels and embeddings for tracking datasets.
- Occlusion and visibility profiles for object-tracking datasets.

**SDP v1.3 @ ML1.** Signed releases and verifiable build provenance, threat
model documentation, data/model cards for shipped artifacts, automated test
evidence packages, and CI gates aligned to ML1 under v1.3.

**AIRCC prioritization workflow on FMV.** Target September 2026.

**Ontology depth.** Hierarchical-taxonomy support in bias and balance
evaluators; ontology drift detection across dataset versions.

**Operational demos.** Deploy and validate DataEval against operational
data, with improvements to dataset exploration and curation ergonomics
informed by real workflows.

---

## Q4 2026 — DataEval v2.0

**v2.0 stable.** Full video support across every evaluator module, validated
on benchmark datasets for classification, object detection, and object
tracking. API freeze mid-December to support downstream app development.

**Video-specific evaluators.** Near-duplicate detection across videos,
clips, and frames; leakage detector for video datasets integrated into the
bias module.

**Documentation and interpretation.** Tutorials, how-to guides, and
interpretation guidance for video-specific results across every module.

**SDP posture.** Maintain v1.3 ML1 compliance; begin gap assessment for ML2
to inform 2027 planning.

---

## Beyond 2026 (directional)

- **SDP v1.3 ML2 compliance** — formal review records, defect density
  tracking, independent verification of release artifacts.
- **FMV depth and scale** — long-video support with streaming embeddings,
  multi-camera and multi-view datasets, real-time evaluation hooks.
- **Synthetic and augmented video** — detection and characterization of
  synthetic content in operational datasets.
- **DataEval v2.1** — FMV evaluator improvements informed by v2.0 field
  feedback.

---

## Success criteria

- All five evaluation modules operational for image and video data,
  validated on benchmark datasets.
- Support for classification, object detection, and object tracking.
- FMV-supported AIRCC prioritization workflow by September 2026.
- DataEval v2.0 production release with comprehensive documentation.
- SDP v1.2 and v1.3 ML1 compliance achieved by end of Q3 2026.
