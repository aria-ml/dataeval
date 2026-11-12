# Explanations

These pages explain what DataEval's evaluators measure, the theory underlying each evaluator and when to apply it. A
page often covers several evaluators that work together. The pages are grouped by the question they help answer, from
the measurements and metadata that evaluators read to the interpretation of their results.

## Image measurements

Statistics that evaluators compute directly from images.

:::{toctree}
:caption: Image measurements
:hidden:

ImageStatistics
BlurStatistics
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](ImageStatistics.md)
  - The pixel, visual, dimension and hash statistics that {func}`.image_stats` computes, and when a value can be
    compared across a dataset.
- - [](BlurStatistics.md)
  - How blur is measured as edge width in pixels for images and videos, and the limits of that measurement.

:::

## Representations and groupings

Learned representations of images, and the groups that evaluators form from those representations.

:::{toctree}
:caption: Representations and groupings
:hidden:

Embeddings
Clustering
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](Embeddings.md)
  - What embeddings are, why evaluators use them and how to create them with extractors.
- - [](Clustering.md)
  - How the clustering that several evaluators depend on works, and what its outputs mean.

:::

## Metadata and labels

How factors and class labels are organized before an evaluator reads them.

:::{toctree}
:caption: Metadata and labels
:hidden:

MetadataLevels
Binning
BinEdges
ExtrinsicFactors
Ontology
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](MetadataLevels.md)
  - Why metadata is kept at the level where it was measured, such as image, detection or video, and how values move
    between levels.
- - [](Binning.md)
  - How continuous factors become discrete categories, and how the choice of bins affects downstream statistics.
- - [](BinEdges.md)
  - How to set bin edges from the question instead of the values, and what a cut costs in the scores computed from it.
- - [](ExtrinsicFactors.md)
  - Which video, platform and time metadata to extract, why a model can learn the differences it records, and where to
    find each factor in a video file.
- - [](Ontology.md)
  - How a domain vocabulary is written down as an ontology, and how DataEval checks labels against it, relates two
    vocabularies and validates the ontology itself.

:::

## Data quality

Samples and labels that reduce the validity of training or evaluation.

:::{toctree}
:caption: Data quality
:hidden:

DataIntegrity
MetadataOutliers
Leakage
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](DataIntegrity.md)
  - Redundant and corrupted samples, anomalies and label noise, and how DataEval detects each.
- - [](MetadataOutliers.md)
  - The methods behind outlier detection over metadata factors, and the evidence behind their defaults.
- - [](Leakage.md)
  - How information that would not be available at deployment contaminates training or evaluation, including through
    repeated evaluation.

:::

## Bias and coverage

Evidence on how the sampled data compare with the intended operational conditions, and on which conditions may be
underrepresented.

:::{toctree}
:caption: Bias and coverage
:hidden:

DatasetBias
DatasetCoverage
Parity
ParityCalibration
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](DatasetBias.md)
  - How the association between class labels and metadata factors and the evenness of sampling are measured, what those
    measures say about the intended operational population, and how an imbalance is addressed.
- - [](DatasetCoverage.md)
  - How coverage and completeness describe the way a dataset occupies the embedding space, what that says about the
    conditions the dataset represents, and how samples are prioritized to fill a gap.
- - [](Parity.md)
  - Tests of whether metadata factors are associated with class labels, for each factor and for each combination of a
    level and a class.
- - [](ParityCalibration.md)
  - The false-positive rates and power of the Parity tests, measured by simulation.

:::

## Attainable performance

What performance a dataset can support.

:::{toctree}
:caption: Attainable performance
:hidden:

PerformanceLimits
FeasibilityEvidence
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](PerformanceLimits.md)
  - Whether a performance requirement is achievable with the available data, whether a specified model has shown that it
    is, and when additional similar samples yield diminishing returns.
- - [](FeasibilityEvidence.md)
  - How to supply evidence to {class}`.Feasibility`, how each kind of evidence is built and how it becomes a
    determination.

:::

## Shift between datasets

Differences between the data that a model was trained on and the operational or evaluation data that it encounters.

:::{toctree}
:caption: Shift between datasets
:hidden:

DistributionShift
Divergence
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](DistributionShift.md)
  - A taxonomy of shift, and how drift and out-of-distribution detectors test for it.
- - [](Divergence.md)
  - How HP divergence measures how far apart two datasets are, and how that distance changes over time.

:::

## Interpreting and trusting results

What a result supports, and how much evidence stands behind it.

:::{toctree}
:caption: Interpreting and trusting results
:hidden:

ActingOnResults
ValidationAndTrust
ValidationEvidence
:::

:::{list-table}
:width: 100%
:widths: 30 70
:header-rows: 0

- - [](ActingOnResults.md)
  - How each category of DataEval output maps to decisions and to the actions available.
- - [](ValidationAndTrust.md)
  - The assumptions behind each evaluator, the levels of validation evidence and the known gaps in that evidence.
- - [](ValidationEvidence.md)
  - For each evaluator and core function, the basis of its evidence, its validation level, its policy levers and its
    contraindications.

:::
