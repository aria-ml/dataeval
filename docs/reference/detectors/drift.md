(drift-ref)=

# Drift Detection

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

## Tutorials

Check out this tutorial to begin using the `Drift Detection` class

{doc}`Drift Detection Tutorial<../../tutorials/notebooks/DriftDetectionTutorial>`

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

### Kolmogorov-Smirnov

The drift detector applies feature-wise two-sample [Kolmogorov-Smirnov] (K-S) tests. For multivariate data, the obtained
p-values for each feature are aggregated either via the [Bonferroni] or the [False Discovery Rate] (FDR) correction.
The Bonferroni correction is more conservative and controls for the probability of at least one false positive. The FDR
correction on the other hand allows for an expected fraction of false positives to occur.

```{eval-rst}
.. autoclass:: dataeval.detectors.DriftKS
   :members:
   :inherited-members:
```

### Maximum Mean Discrepancy

The [Maximum Mean Discrepancy] (MMD) detector is a kernel-based method for multivariate 2 sample testing. The MMD is
a distance-based measure between 2 distributions *p* and *q* based on the mean embeddings $\mu_{p}$ and $\mu_{q}$
in a reproducing kernel Hilbert space $F$:

$$
MMD(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}
$$

We can compute unbiased estimates of $MMD^2$ from the samples of the 2 distributions after applying the kernel trick.
We use by default a radial basis function kernel, but users are free to pass their own kernel of preference to the detector.
We obtain a $p$-value via a permutation test on the values of $MMD^2$.

```{eval-rst}
.. autoclass:: dataeval.detectors.DriftMMD
   :members:
   :inherited-members:
```

### Classifier Uncertainty

The classifier uncertainty drift detector aims to directly detect drift that is likely to effect the performance of a model
of interest. The approach is to test for change in the number of instances falling into regions of the input space on which
the model is uncertain in its predictions. For each instance in the reference set the detector obtains the model's prediction
and some associated notion of uncertainty. The same is done for the test set and if significant differences in uncertainty
are detected (via a [Kolmogorov-Smirnov] test) then drift is flagged. The detector's reference set should be disjoint from
the model's training set (on which the model's confidence may be higher).

```{eval-rst}
.. autoclass:: dataeval.detectors.DriftUncertainty
   :members:
   :inherited-members:
```

### GaussianRBF

The GaussianRBF class implements a Gaussian kernel, also known as a [radial basis function] (RBF) kernel. It is used
to construct a covariance matrix for gaussian processes and is the default kernel used in the MMD drift detection test.

```{eval-rst}
.. autoclass:: dataeval.detectors.GaussianRBF
   :members:
```

### LastSeenUpdate

```{eval-rst}
.. autoclass:: dataeval.detectors.LastSeenUpdate
   :members:
   :inherited-members:
```

### ReservoirSamplingUpdate

```{eval-rst}
.. autoclass:: dataeval.detectors.ReservoirSamplingUpdate
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
