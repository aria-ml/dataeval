# _Cramér-von Mises_

The {term}`Cramér-von Mises<Cramér-von Mises (CVM) Test>` is a non-parametric
method used for detecting drift by comparing two empirical distributions. For
two distributions $F(z)$ and $F_{\textrm{ref}}(z)$, the CVM test statistic is
calculated as:

$$
W = \sum_{z\in k} \left| F(z) - F_{\textrm{ref}}(z) \right|^2
$$

where $k$ represents the {term}`joint sample<Joint Sample>`. The CVM test is
particularly effective in detecting shifts in higher-order moments, such as
changes in {term}`variance<Variance>`, by leveraging the full joint sample.

When applied to multivariate data, the CVM test is conducted separately for each
feature, and the resulting p-values are aggregated using either the [Bonferroni]
or {term}`False Discovery Rate (FDR)` correction. The
{term}`Bonferroni correction<Bonferroni Correction>` controls the probability of
at least one false positive, making it more conservative, while the FDR
correction allows for a controlled proportion of false positives.

**Key characteristics:**

- **Sensitivity**: Higher sensitivity to subtle distributional shifts across
  the entire range
- **Weighting**: Gives equal weight to differences across the entire
  distribution

- **Power**: Generally more powerful than KS for detecting variance changes
- **Granularity**: Integrates over all points rather than taking maximum
  difference

**When to use:**

- **Camera/lighting condition changes** (subtle exposure, white balance drift)
- Detecting subtle shifts in higher-order moments (variance, skewness)
- When distribution changes are spread across multiple regions
- For continuous variables with fine-grained distributional changes
- When you need higher statistical power than KS

**Limitations:**

Computationally more intensive than KS (though usually negligible for most ML use
cases) and less intuitive to visualize or explain to non-technical stakeholders.

[bonferroni]: https://en.wikipedia.org/wiki/Bonferroni_correction
