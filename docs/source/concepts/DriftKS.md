# _Kolmogorov-Smirnov_

The {term}`Kolmogorov-Smirnov test<Kolmogorov-Smirnov (K-S) test>` is another
widely used non-parametric test for detecting {term}`drift<Drift>`. It measures
the maximum distance between two empirical distributions, $F(z)$ and
$F_{\textrm{ref}}(z)$:

$$
\textrm{KS} = \sup_{x} \left| F(z) - F_{\textrm{ref}}(z) \right|
$$

where $\sup_{x}$ is the supremum of the set of distances. The KS test is
particularly useful for detecting differences in the distribution's shape, such
as shifts in location or scale.

Similar to the CVM test, when dealing with multivariate data, the KS test is
applied to each feature separately. The resulting p-values are then aggregated
using either the [Bonferroni] or {term}`False Discovery Rate (FDR)` correction.
The {term}`Bonferroni correction<Bonferroni Correction>` controls the
probability of at least one false positive, making it more conservative, while
the FDR correction allows for a controlled proportion of false positives.

**Key characteristics:**

- **Sensitivity**: Most sensitive to differences in the middle of the
  distribution
- **Test type**: Two-sided test (detects both shifts in location and scale)
- **Distribution-free**: Makes no assumptions about the underlying distributions
- **Power**: Good general-purpose test, though can be less powerful for tail
  differences

**When to use:**

- **Image pixel statistics** (brightness, contrast per RGB channel)
- General-purpose drift detection across various feature types
- When you need a well-understood, widely accepted statistical test
- When drift is expected in the central portions of the distribution
- As a baseline comparison for other methods

**Limitations:**

Blind to the edges. It struggles to detect changes that occur in the tails (outliers)
and can be overly sensitive to sample size, leading to "false alarms" with very large
datasets.

[bonferroni]: https://en.wikipedia.org/wiki/Bonferroni_correction
