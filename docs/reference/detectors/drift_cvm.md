(driftcvm-ref)=
# Drift CVM

Drift refers to the phenomenon where the statistical properties of the data change over time. It occurs when the underlying
distribution of the input features or the target variable (what the model is trying to predict) shifts, leading to a discrepancy
between the training data and the real-world data the model encounters during deployment.

Through concepts examined in the NeurIPS 2019 paper [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953),
we can utilize various methods in order to determine if drift is detected. For high-dimensional
data, we typically want to reduce the dimensionality before performing tests against the dataset. To do so, we incorporate Untrained
AutoEncoders (UAE) and Black-Box Shift Estimation ([BBSE]) predictors using the classifier's softmax outputs as out-of-the box
preprocessing methods and note that [Principal Component Analysis] can also be easily implemented using [scikit-learn].
Preprocessing methods which do not rely on the classifier will usually pick up drift in the input data, while [BBSE] focuses
on label shift.

## How-To Guides

Check out this **how to** to begin using the `Drift Detection` class

{doc}`Drift Detection Tutorial<../../how_to/notebooks/DriftDetectionTutorial>`

## DataEval API

### Cramér-von Mises (CVM)

The CVM drift detector is a non-parametric drift detector, which applies feature-wise two-sample Cramér-von Mises (CVM) tests.
For two empirical distributions $F(z)$ and $F_{ref}(z)$, the CVM test statistic is defined as

$$
W = \sum_{z\in k} \left| F(z) - F_{ref}(z) \right|^2
$$

where $k$ is the joint sample. The CVM test is an alternative to the [Kolmogorov-Smirnov] (K-S) two-sample test, which
uses the maximum distance between two empirical distributions $F(z)$ and $F_{ref}(z)$. By using the full joint
sample, the CVM can exhibit greater power against shifts in higher moments, such as variance changes.

For multivariate data, the detector applies a separate CVM test to each feature, and the p-values obtained for each feature
are aggregated either via the [Bonferroni] or the [False Discovery Rate] (FDR) correction. The Bonferroni correction is more
conservative and controls for the probability of at least one false positive. The FDR correction on the other hand allows for
an expected fraction of false positives to occur. As with other univariate detectors such as the [Kolmogorov-Smirnov] detector,
for high-dimensional data, we typically want to reduce the dimensionality before computing the feature-wise univariate FET
tests and aggregating those via the chosen correction method.

```{eval-rst}
.. autoclass:: dataeval.detectors.DriftCVM
   :members:
   :inherited-members:
```


[bbse]: https://arxiv.org/abs/1802.03916
[bonferroni]: https://mathworld.wolfram.com/BonferroniCorrection.html
[drift_ref]: https://arxiv.org/abs/1802.03916
[false discovery rate]: http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf
[kolmogorov-smirnov]: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
[maximum mean discrepancy]: http://jmlr.csail.mit.edu/papers/v13/gretton12a.html
[principal component analysis]: https://en.wikipedia.org/wiki/Principal_component_analysis
[radial basis function]: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
[scikit-learn]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
