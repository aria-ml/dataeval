# Drift

## What is drift?

{term}`Drift` refers to the phenomenon where the statistical properties of data change
over time, leading to discrepancies between the data a model was trained on and
the data it encounters during deployment. This can significantly degrading the
performance of {term}`machine learning<Machine Learning (ML)>` models, as the
assumptions made during training may no longer hold in real-world scenarios.

### _Formal Definition and Types of Drift_

In the context of {term}`supervised learning<Supervised Learning>`, where a
model is trained to predict the conditional probability $P(Y|X)$—with $X$
representing input features and $Y$ representing the target variable—drift
occurs when the joint distribution $P(X, Y)$ changes between the training and
deployment phases. Specifically, drift is observed when the joint distribution
$P_t(X,Y)$ during training differs from the joint distribution $P_d(X,Y)$
during deployment:

$$
P_t(X, Y) \neq P_d(X, Y)
$$

The joint distribution $P(X, Y)$ can be decomposed into two equivalent forms:

- As the product of the posterior probability and the evidence:
  $P(X, Y) = P(Y|X)P(X)$
- As the product of the likelihood and the prior: $P(X, Y) = P(X|Y)P(Y)$

Different types of drift can be identified by analyzing which components of
these decompositions have changed.

### _Covariate Shift_

Covariate Shift (also known as population shift or virtual drift) occurs when
the conditional probability of the target given the input, $P(Y|X)$, remains
unchanged, but the distribution of the input features, $P(X)$, changes between
training and deployment:

$$
P_t(Y|X) = P_d(Y|X) \quad \text{but} \quad P_t(X) \neq P_d(X)
$$

This type of shift often arises due to environmental variability, sensor
degradation, or biased sampling during training. Covariate shift can lead to
poor model performance if the model has not seen certain regions of the input
space during training (i.e., "blind spots").

### _Label Shift_

{term}`Label shift<Label Shift>` (also known as prior-probability shift or
target shift) occurs when the conditional probability of the input given the
output, $P(X|Y)$, remains the same, but the prior distribution of the target
variable, $P(Y)$, shifts between training and deployment:

$$
P_t(X|Y) = P_d(X|Y) \quad \text{but} \quad P_t(Y) \neq P_d(Y)
$$

Label shift can result from biased sampling, particularly with rare targets,
leading to poor model calibration. While label shift can be related to
covariate shift, it specifically focuses on changes in the distribution of the
target variable.

### _Concept Drift_

{term}`Concept drift<Concept Drift>` (also known as posterior-probability shift
or real drift) occurs when the input distribution, $P(X)$, remains stable, but
the conditional probability of the target given the input, $P(Y|X)$, changes:

$$
P_t(X) = P_d(X) \quad \text{but} \quad P_t(Y|X) \neq P_d(Y|X)
$$

Concept drift causes the same input features to correspond to different outputs
over time. This can happen due to changes in the underlying process that
generates the data, such as evolving disease patterns in medical diagnosis or
changes in the characteristics of targets in machine
{term}`classification<Classification>` tasks. Concept drift can be driven by
changes in the likelihood $P(X|Y)$, the prior $P(Y)$, or both.

<!--
### _Semantic Drift_

Semantic Drift refers to a situation where new classes appear in the testing or
deployment  phase that were not present in the training set. This type of drift
presents a unique challenge because the model has never encountered these new
classes before, making it difficult to predict their behavior accurately.
Semantic drift often requires the development of mechanisms to identify and
adapt to new classes dynamically, such as using open-set recognition techniques
or incorporating mechanisms for continuous learning.
-->

### _Interrelations Between Drift Types_

It's important to note that these forms of {term}`drift<Drift>` are not
mutually exclusive. For instance, covariate shift can lead to
{term}`Label shift<Label Shift>`, and both covariate and label shifts can
contribute to {term}`concept drift<Concept Drift>`. As observed in the
[literature](#references), it is common for covariate shift and concept
drift to occur simultaneously. However, it is challenging, if not impossible,
to study dataset shifts that result from isolated changes in the likelihood
$P(X|Y)$ alone.

### _Practical Implications_

Understanding these different forms of {term}`drift<Drift>` is crucial for
maintaining robust, reliable and generalizable
{term}`machine learning<Machine Learning (ML)>` models over time. Techniques
such as Untrained AutoEncoders (UAE) and
{term}`black box shift estimation<Black-Box Shift Estimation (BBSE)>` can be
applied as preprocessing methods to detect these shifts.
{term}`Dimensionality reduction<Dimensionality Reduction>` techniques, like
{term}`principal component analysis<Principal Component Analysis (PCA)>`, are
also often used to simplify the problem in high-dimensional datasets. By
carefully monitoring for drift and applying appropriate corrective measures, we
can ensure that our models continue to perform well in dynamic, real-world
environments.

## When to monitor for drift

Once a model has been approved for operation/deployment, monitoring for
drift should begin. Regardless of whether the model has been
deployed in a contiunous or discrete manner, all new data that the model
digests should be analyzed for drift.

## Theory behind drift detection

### _Cramér-von Mises_

The {term}`Cramér-von Mises<Cramér-von Mises (CVM) Drift Detection>` test is a
non-parametric method used for detecting drift by comparing two
empirical distributions. For two distributions $F(z)$ and $F_{ref}(z)$, the CVM
test statistic is calculated as:

$$
W = \sum_{z\in k} \left| F(z) - F_{ref}(z) \right|^2
$$

where $k$ represents the {term}`joint sample<Joint Sample>`.
The CVM test is particularly effective in detecting shifts in higher-order
moments, such as changes in {term}`variance<Variance>`, by leveraging the full
joint sample.

When applied to multivariate data, the CVM test is conducted separately for
each feature, and the resulting p-values are aggregated using either the
[Bonferroni] or {term}`False Discovery Rate (FDR)` correction. The
{term}`Bonferroni correction<Bonferroni Correction>` controls the probability
of at least one false positive, making it more conservative, while the FDR
correction allows for a controlled proportion of false positives.

### _Kolmogorov-Smirnov_

The {term}`Kolmogorov-Smirnov test<Kolmogorov-Smirnov (K-S) test>` is another
widely used non-parametric test for detecting {term}`drift<Drift>`. It measures
the maximum distance between two empirical distributions, $F(z)$ and
$F_{ref}(z)$:

$$
KS = \sup_{x} \left| F(z) - F_{ref}(z) \right|
$$

where $\sup_{x}$ is the supremum of the set of distances. The KS test is
particularly useful for detecting differences in the distribution's shape, such
as shifts in location or scale.

Similar to the CVM test, when dealing with multivariate data, the KS test is
applied to each feature separately. The resulting p-values are then aggregated
using either the Bonferroni or FDR correction, depending on whether the
priority is to minimize false positives or to allow a controlled number of
them.

### _Maximum Mean Discrepancy_

{term}`Maximum Mean Discrepancy (MMD) Drift Detection` is a kernel-based method
for comparing two distributions by calculating the distance between their mean
{term}`embeddings<Embeddings>` in a reproducing kernel Hilbert space (RKHS).
The MMD test statistic is defined as:

$$
MMD(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}
$$

where $\mu_{p}$ and $\mu_{q}$ are the mean embeddings of distributions _p_ and
_q_ in the RKHS. The MMD test is particularly useful for detecting complex,
multivariate distributional differences. Unbiased estimates of $MMD^2$ can be
obtained using the [kernel trick], and a permutation test is used to obtain the
{term}`p-value<P-Value>`.

A common choice for the kernel is the [radial basis function] (RBF) kernel,
though other kernels can be used depending on the application.

### _Classifier Uncertainty_

Classifier Uncertainty as a {term}`drift<Drift>` detection method focuses on
changes in the model's uncertainty across different datasets. This approach is
particularly relevant when the goal is to detect drift that could impact the
performance of a model in production. The method works by comparing the
distribution of prediction uncertainties (e.g., softmax outputs) between a
reference dataset and a test dataset. Significant differences, typically
detected via a KS test, indicate potential drift.

This method is especially useful when the reference set is distinct from the
training set, as it helps detect shifts in regions where the model's
predictions are less confident.

## Detecting drift

DataEval is a comprehensive data analysis and monitoring library that provides
several classes specifically designed for drift detection. These classes
implement the theoretical concepts discussed above to help detect drift in
datasets efficiently.

DataEval's drift detection classes are:

- **{func}`.DriftCVM`**: Implements the Cramér-von Mises (CVM) test for
  feature-wise drift detection.
- **{func}`.DriftKS`**: Implements the
  Kolmogorov-Smirnov test for detecting feature-wise distributional shifts.
- **{func}`.DriftMMD`**: Utilizes the Maximum Mean Discrepancy (MMD) test to
  detect drift in multivariate data using kernel methods.
- **{func}`.DriftUncertainty`**: Detects drift by analyzing changes in the
  model's uncertainty across datasets.

To see how these detectors work in practice, refer to our
[Monitoring Guide](../tutorials/Data_Monitoring.ipynb),
where you can explore real-world examples of drift detection using DataEval.

## References

1. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
a practical and powerful approach to multiple testing. Journal of the Royal
statistical society: series B (Methodological), 57(1), 289-300.
<https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
2. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
(2012). A kernel two-sample test. The Journal of Machine Learning Research,
13(1), 723-773. <https://jmlr.csail.mit.edu/papers/v13/gretton12a.html>
3. Huyen, C. (2022). Designing machine learning systems. O'Reilly Media, Inc.
<https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/>
4. Lipton, Z., Wang, Y. X., & Smola, A. (2018, July). Detecting and correcting
for label shift with black box predictors. In International conference on
machine learning (pp. 3122-3130). PMLR. <https://arxiv.org/abs/1802.03916>
5. Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An
empirical study of methods for detecting dataset shift. Advances in Neural
Information Processing Systems, 32. <https://arxiv.org/abs/1810.11953>
6. Yuan, L., Li, H., Xia, B., Gao, C., Liu, M., Yuan, W., & You, X. (2022,
July). Recent Advances in Concept Drift Adaptation Methods for Deep Learning.
In IJCAI (pp. 5654-5661). <https://www.ijcai.org/proceedings/2022/0788.pdf>

[bonferroni]: https://en.wikipedia.org/wiki/Bonferroni_correction
[kernel trick]: https://en.wikipedia.org/wiki/Kernel_method
[radial basis function]: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
