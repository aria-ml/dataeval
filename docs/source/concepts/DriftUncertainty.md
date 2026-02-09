<!-- # _Classifier Uncertainty_

Classifier Uncertainty as a {term}`drift<Drift>` detection method focuses on
changes in the model's uncertainty across different datasets. This approach is
particularly relevant when the goal is to detect drift that could impact the
performance of a model in production. The method works by comparing the
distribution of prediction uncertainties (e.g., softmax outputs) between a
reference dataset and a test dataset. Significant differences, typically
detected via a KS test, indicate potential drift.

This method is especially useful when the reference set is distinct from the
training set, as it helps detect shifts in regions where the model's
predictions are less confident. -->

# Uncertainty-Based Drift Detection

This page explains uncertainty-based drift detection (UBDD), a method that
monitors changes in model prediction confidence to identify drift that is likely
to degrade performance. For examples using UBDD in practice, see our
[Monitoring Guide](../notebooks/tt_monitor_shift.md).

## What is uncertainty-based drift detection?

Uncertainty-based drift detection (UBDD) is a method that identifies dataset
{term}`drift<Drift>` by monitoring changes in the distribution of model
confidence scores. Unlike approaches that directly examine input feature
distributions, UBDD focuses on the model's output behavior - specifically how
prediction confidence values disperse across possible outcomes compared to an
established reference distribution. Higher {term}`entropy<Entropy>` in these
confidence distributions (more uniform spread across classes) indicates greater
uncertainty, which may signal that incoming data differs significantly from the
reference data.

So this method compares how uncertain a classifier is about its predictions on a
reference dataset versus a test dataset. When the model becomes significantly
more uncertain on new data—as determined by a statistical test—drift is flagged,
signaling that the new data may fall into regions where the model's predictions
are less reliable.

## Why use uncertainty-based drift detection?

Typical feature-based drift detectors (like {class}`.DriftUnivariate`) flag any
change in the input distribution $P(X)$, regardless of whether those changes
affect model predictions. This can produce false alarms when irrelevant features
change (like background pixels or lighting) while missing performance degradation
when the relationship between inputs and outputs shifts but the overall feature
distribution appears stable.

UBDD focuses specifically on whether the model encounters data in its
uncertainty regions, making it a **performance-oriented** detector that catches
drift affecting predictions while ignoring changes that don't impact model
behavior. Since the model's uncertainty inherently reflects its learned decision
boundaries, UBDD provides a model-aware signal that connects directly to
prediction quality.

## How does it work?

Uncertainty-based drift detection requires a way to quantify how uncertain a
classifier is about its predictions. DataEval, following
[Alibi-Detect](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/modeluncdrift.html),
uses **entropy** as its uncertainty measure.

For each image, a trained classifier produces class probabilities—for example,
`[0.7, 0.2, 0.1]` for a three-class problem. Entropy measures how "spread out"
these probabilities are:

$$
H(p) = -\sum_{i=1}^{k} p_i \log(p_i)
$$

where $p_i$ is the probability assigned to class $i$ and $k$ is the number of
classes.

- **Low entropy** (confident): `[0.95, 0.03, 0.02]` → low entropy, model is
  confident
- **High entropy** (uncertain): `[0.4, 0.35, 0.25]` → high entropy, model is
  uncertain

The detector compares entropy distributions between reference and test datasets
using a selected univariate statistical test such as
{term}`Kolmogorov-Smirnov test<Kolmogorov-Smirnov (K-S) test>`. If the test
data has significantly higher entropy, drift is flagged.

While DataEval uses entropy, other uncertainty measures exist.
[Sethi and Kantardzic (2017)](#references) originally proposed monitoring
whether predictions fall within a confidence margin. The choice of measure
affects sensitivity, but the core principle remains: monitoring changes in model
confidence to identify potentially problematic drift.

## When to use it

Use the guidance below to determine whether uncertainty-based drift detection is
appropriate for your situation:

**✓ Use UBDD when:**

- You have a **trained classifier** that outputs class probabilities.
- You're monitoring for **{term}`concept drift<Concept Drift>`** ($P(Y|X)$
  changes) that could degrade performance.
- Ground truth labels are **unavailable or delayed**, making direct performance
  monitoring impractical.
- You want to avoid false alarms from **irrelevant feature changes**.

**✗ Don't use UBDD when:**

- You want to detect **all distributional changes** in inputs, not just
  performance-relevant ones—use feature-based detectors like {class}`.DriftMMD`.
- You're using a **regression model** (UBDD requires classification
  probabilities).
- You don't have a trained model yet (UBDD requires model predictions).

**Note:** UBDD may miss drift if the model remains confident on new data even
when predictions are wrong. The detector's reference set should be separate from
the model's training set, as models are typically more confident on training
data.

## Detecting uncertainty-based drift

DataEval implements uncertainty-based drift detection through the
{class}`.DriftUncertainty` class. This class requires a PyTorch classifier that
outputs class probabilities and compares prediction entropy distributions
between reference and test datasets.

To see how {class}`.DriftUncertainty` works in practice, refer to our
[Monitoring Guide](../notebooks/tt_monitor_shift.md).

## References

1. Sethi, T. S., & Kantardzic, M. (2017). On the reliable detection of concept
   drift from streaming unlabeled data. _Expert Systems with Applications_, 82,
   77-99. <https://arxiv.org/abs/1704.00023>
2. Van Looveren, A., et al. (2019-2024). Alibi Detect: Algorithms for outlier,
   adversarial and drift detection. _Seldon Technologies_.
   <https://docs.seldon.io/projects/alibi-detect/>
