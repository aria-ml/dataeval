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

[bonferroni]: https://en.wikipedia.org/wiki/Bonferroni_correction
