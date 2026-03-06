# DataEval Roadmap

**Last updated:** February 2026

---

## Timeline overview

```text
                    Q1 2026  |    Q2 2026    |    Q3 2026     |   Q4 2026
                    Feb  Mar   Apr  May  Jun   Jul  Aug   Sep  Oct  Nov   Dec
                   |----|----|-----|----|----|-----|----|-----|----|----|-----|

Library
  v1.0 stable      [####]
  v1.1                  [###############]
  v2.0rc (video)                                                         [#####]

Video adaptation
  Phase 1          [######]
  Phase 2                 [########]
  Phase 3                          [####################]
  Phase 4                                   [############################]
  Phase 5                  [################################################]

App (containerized)
  Phase 1          [########]
  Phase 2                   [#####################]
  Phase 3                                         [#############################]

```

## Library

### DataEval v1.0

**Target:** end of February 2026

- [x] Quality module: outlier detection, duplicate detection, core set selection
- [x] Performance module: Bayes Error Rate (BER), Upper-bound Average Precision (UAP), performance projection
- [x] Bias module: parity, diversity, balance, leakage detection
- [x] Shifts module: out-of-distribution (OOD) detection, drift analysis
- [x] API freeze mid-February to support app development without breaking changes
- [x] v1.0 stable release

### DataEval v1.1

**Target:** May 2026

- [ ] Scope module: coverage, completeness
- [ ] Incorporate additional capabilities as developed

### DataEval v2.0 — full video support

**Target:** December 2026

- [ ] API freeze mid-December to support app development without breaking changes
- [ ] v2.0 stable release

## Video adaptation

Extending DataEval to full motion video (FMV).
Most evaluators run on embeddings, metadata, and statistics — the strategy focuses on video-appropriate inputs
and thorough interpretation documentation rather than rewriting tool algorithms.

**Success criteria:**

- All five evaluation modules operational for video data, validated on benchmark datasets
- Support for classification, object detection, and object tracking task types
- FMV-supported AIRCC prioritization workflow by September 2026 🔥
- DataEval 2.0 production release with comprehensive documentation

### Phase 1 — Foundation and infrastructure

**Target:** mid-March 2026

- [ ] `VideoClassificationDataset`, `VideoObjectDetectionDataset`, `VideoObjectTrackingDataset` classes
- [ ] Key frame selection algorithm (temporally independent frames)
- [ ] Frame-level embeddings using existing image encoders
- [x] Video-level embeddings — new `VideoEmbeddings` class (VideoMAE or similar)
- [ ] Clipping and variable framerate normalization
- [ ] Metadata extraction: compression, resolution, frame rate, color space
- [ ] Video-aware train/test splitting (assigns whole videos to prevent frame-level leakage)

### Phase 2 — Video statistics

**Target:** April 2026

- [ ] FMV statistics module
  - [ ] Time stats: clip duration, frame count, frame rate
  - [ ] Motion stats: optical flow magnitude, variation, object motion
  - [ ] Quality stats: compression level, motion blur, occlusion metrics
  - [ ] Aggregated frame stats: brightness, contrast, color entropy
- [ ] Optical flow maps for motion analysis and embedding

### Phase 3 — Module adaptation

**Target:** August 2026

- [ ] Quality module: outlier detection, duplicate detection, core set selection for video
- [ ] Shifts module: OOD detection, divergence, drift detection for video *(drift will likely require modification)*
- [ ] Bias module validation: parity, diversity, balance with video data
- [ ] Scope module validation: concept coverage and completeness with video data
- [ ] Performance module research: BER, UAP, performance projection for video

### Phase 4 — New video-specific evaluators

**Target:** November 2026

- [ ] Leakage detector for video datasets
- [ ] Near-duplicate detector: near-duplicate videos, clips, and frames within a dataset
- [ ] Integrate leakage detector into bias module
- [ ] Integrate near-duplicate detector into quality module
- [ ] AIRCC prioritization workflow adapted for FMV — target: September 2026 🔥

### Phase 5 — Documentation and deployment

**Target:** Decemember 2026

- [ ] Update all current documentation to include video data information
- [ ] How-to guides and tutorials for each evaluation module with video examples
- [ ] Interpretation guidance for video-specific results and metrics
- [ ] Additional video-specific workflow examples

## App: containerized workflows

The containerized application exposes DataEval capabilities as automated, non-interactive workflows.
Users provide dataset paths and a configuration file; the container produces data reports, visualizations,
and go/no-go recommendations.

**Supported input formats:** HuggingFace, COCO, YOLO

### Phase 1 — Alpha prototype and initial workflow

**Target:** March 2026

- [x] DataEval App V0.0alpha
  - [x] Docker/Podman container definitions for local deployment
  - [x] Local data ingestion
- [x] Data Validation Workflow
  - The container pulls the reference dataset, uses DataEval to compute embeddings, extract metadata,
  run duplicate/outlier/leakage checks, and generates a "Data Card".
- [ ] Data Cleaning Workflow
  - The container reads raw data and references the data card/cached rules. It utilizes DataEval's core utility
  functions to drop erroneous images and generates a clean dataset.

### Phase 2 — Additional workflows

**Target:** July 2026

- [ ] Monitoring Workflow
  - The app streams incoming data, passes it through the model (if checking prediction drift) or
  DataEval (if checking data drift/OOD). Crucially, it fetches the baseline distributions from the Cache Store
  (seeded by the Validator) to do lightning-fast comparative analysis without needing the whole reference dataset in memory.
  Triggers an alert or pushes data to a dashboard (e.g. Prometheus/Grafana)
- [ ] Prioritization Workflow
  - This container ingests unlabeled incoming data, the current model, and cached metadata stats.
  It uses DataEval to find images that represent high uncertainty (model is confused), high diversity
  (furthest from reference dataset clusters), or underrepresented metadata factors.
- [ ] Integration and testing with real-world field operational datasets
- [ ] Full Motion Video (FMV) feasibility assessment — informs Phase 3 and video roadmap

### Phase 3 — FMV expansion

**Target:** February 2027

- [ ] FMV integration (contingent on Phase 2 feasibility findings)
- [ ] Integration and testing with real-world field operational FMV datasets
