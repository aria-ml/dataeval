# Drift

```{toctree}
:maxdepth: 1
:hidden:

DriftCVM
DriftKS
DriftMMD
DriftMVDC
DriftUncertainty
```

## What is drift?

{term}`Drift` refers to the phenomenon where the statistical properties of data
change over time, leading to discrepancies between the data a model was trained
on and the data it encounters during deployment. This can significantly degrade
the performance of {term}`machine learning<Machine Learning (ML)>` models, as
the assumptions made during training may no longer hold in real-world scenarios.

### _Formal Definition and Types of Drift_

In the context of {term}`supervised learning<Supervised Learning>`, where a
model is trained to predict the conditional probability $P(Y|X)$—with $X$
representing input features and $Y$ representing the target variable—drift
occurs when the joint distribution $P(X, Y)$ changes between the training and
deployment phases. Specifically, drift is observed when the joint distribution
$P_t(X,Y)$ during training differs from the joint distribution $P_d(X,Y)$ during
deployment:

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

{term}`Covariate shift<Covariate Shift>` (also known as population shift or
virtual drift) occurs when the conditional probability of the target given the
input, $P(Y|X)$, remains unchanged, but the distribution of the input features,
$P(X)$, changes between training and deployment:

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
leading to poor model calibration. While label shift can be related to covariate
shift, it specifically focuses on changes in the distribution of the target
variable.

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

It's important to note that these forms of {term}`drift<Drift>` are not mutually
exclusive. For instance, covariate shift can lead to
{term}`label shift<Label Shift>`, and both covariate and label shifts can
contribute to {term}`concept drift<Concept Drift>`. As observed in the
[literature](#references), it is common for covariate shift and concept drift to
occur simultaneously. However, it is challenging, if not impossible, to study
dataset shifts that result from isolated changes in the likelihood $P(X|Y)$
alone.

### _Practical Implications_

Understanding these different forms of {term}`drift<Drift>` is crucial for
maintaining robust, reliable and generalizable
{term}`machine learning<Machine Learning (ML)>` models over time. Techniques
such as Untrained AutoEncoders (UAE) [\[5\]](#references) and
{term}`black box shift estimation<Black-Box Shift Estimation (BBSE)>` can be
applied as preprocessing methods to detect these shifts.
{term}`Dimensionality reduction<Dimensionality Reduction>` techniques, like
{term}`principal component analysis<Principal Component Analysis (PCA)>`, are
also often used to simplify the problem in high-dimensional datasets. By
carefully monitoring for drift and applying appropriate corrective measures, we
can ensure that our models continue to perform well in dynamic, real-world
environments.

## When to monitor for drift

Once a model has been approved for operation/deployment, monitoring for drift
should begin. Regardless of whether the model has been deployed for batch
processing or real-time inference, all new data that the model digests should be
analyzed for drift.

## Detecting drift

DataEval is a comprehensive data analysis and monitoring library that provides
several classes specifically designed for drift detection. These classes
implement the theoretical concepts discussed in the next section to help detect
drift in datasets efficiently.

DataEval's drift detection classes are:

- **{class}`.DriftCVM`**: Implements the Cramér-von Mises (CVM) test for
  feature-wise drift detection.
- **{class}`.DriftKS`**: Implements the Kolmogorov-Smirnov test for detecting
  feature-wise distributional shifts.
- **{class}`.DriftMMD`**: Utilizes the Maximum Mean Discrepancy (MMD) test to
  detect drift in multivariate data using kernel methods.
- **{class}`.DriftUncertainty`**: Detects drift by analyzing changes in the
  model's uncertainty across datasets.
- **{class}`.DriftMVDC`**: Detects drift by using {term}`AUROC<AUROC>` to detect
  when incoming data are distinguishable from reference data.

To see how these detectors work in practice, refer to our
[Monitoring Guide](../notebooks/tt_monitor_shift.ipynb), where you can explore
real-world examples of drift detection using DataEval.

## Understanding the drift detectors

### _Cramér-von Mises_

The Cramér-von Mises test is a non-parametric method for detecting drift by
measuring the sum of squared differences between two empirical distributions. It
excels at detecting shifts in higher-order moments like variance, and when
applied to multivariate data, it operates on each feature separately with
p-values aggregated using either Bonferroni or FDR correction techniques.
[Read more...](DriftCVM.md)

### _Kolmogorov-Smirnov_

The Kolmogorov-Smirnov test measures the maximum distance between two empirical
distributions to detect drift, making it effective for identifying shifts in
distribution shape, location, or scale. When applied to multivariate data, it
analyzes each feature independently, with resulting p-values aggregated using
either Bonferroni or FDR correction methods. [Read more...](DriftKS.md)

### _Maximum Mean Discrepancy_

Maximum Mean Discrepancy is a kernel-based method that detects drift by
measuring the distance between mean embeddings of two distributions in a
reproducing kernel Hilbert space. This approach excels at identifying complex
multivariate distributional differences, typically using the RBF kernel and
permutation tests to determine statistical significance.
[Read more...](DriftMMD.md)

### _Classifier Uncertainty_

Classifier Uncertainty drift detection monitors changes in a model's predictive
confidence by comparing uncertainty distributions (such as softmax outputs)
between reference and test datasets. Significant differences, typically
identified using a KS test, signal potential drift that could impact model
performance, making this approach particularly valuable for detecting shifts in
regions where the model is less confident. [Read more...](DriftUncertainty.md)

### _Multi-Variate Domain Classifier_

The Domain Classifier detects multivariate drift by training a machine learning
classifier to discriminate between reference and analysis data distributions,
with the resulting AUROC score indicating drift severity (0.5 suggesting no
drift, values approaching 1.0 indicating significant drift). This method excels
at detecting subtle shifts in joint feature distributions that might be missed
by univariate approaches, making it particularly effective for complex,
non-linear relationships in data. [Read more...](DriftMVDC.md)

## References

1. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a
   practical and powerful approach to multiple testing. Journal of the Royal
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
