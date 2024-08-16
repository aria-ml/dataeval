(driftmmd-ref)=
# Drift MMD

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

[bbse]: https://arxiv.org/abs/1802.03916
[bonferroni]: https://mathworld.wolfram.com/BonferroniCorrection.html
[drift_ref]: https://arxiv.org/abs/1802.03916
[false discovery rate]: http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf
[kolmogorov-smirnov]: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
[maximum mean discrepancy]: http://jmlr.csail.mit.edu/papers/v13/gretton12a.html
[principal component analysis]: https://en.wikipedia.org/wiki/Principal_component_analysis
[radial basis function]: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
[scikit-learn]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
