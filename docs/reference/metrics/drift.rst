.. _drift_ref:

.. _BBSE: https://arxiv.org/abs/1802.03916
.. _Kolmogorov-Smirnov: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
.. _Bonferroni: https://mathworld.wolfram.com/BonferroniCorrection.html
.. _False Discovery Rate: http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf
.. _Principal Component Analysis: https://en.wikipedia.org/wiki/Principal_component_analysis
.. _Maximum Mean Discrepancy: http://jmlr.csail.mit.edu/papers/v13/gretton12a.html
.. _radial basis function: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
.. _scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

===============
Drift Detection
===============

Drift refers to the phenomenon where the statistical properties of the data change over time. It occurs when the underlying
distribution of the input features or the target variable (what the model is trying to predict) shifts, leading to a discrepancy
between the training data and the real-world data the model encounters during deployment.

Through concepts examined in the NeurIPS 2019 paper `Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift
<https://arxiv.org/abs/1810.11953>`_, we can utilize various methods in order to determine if drift is detected. For high-dimensional
data, we typically want to reduce the dimensionality before performing tests against the dataset. To do so, we incorporate Untrained
AutoEncoders (UAE) and Black-Box Shift Estimation (`BBSE`_) predictors using the classifier's softmax outputs as out-of-the box
preprocessing methods and note that `Principal Component Analysis`_ can also be easily implemented using `scikit-learn`_.
Preprocessing methods which do not rely on the classifier will usually pick up drift in the input data, while `BBSE`_ focuses
on label shift.

---------
Tutorials
---------

Check out this tutorial to begin using the ``Drift Detection`` class

:doc:`Drift Detection Tutorial<../../tutorials/notebooks/DriftDetectionTutorial>`

--------
DAML API
--------

Cramér-von Mises
^^^^^^^^^^^^^^^^

The CVM drift detector is a non-parametric drift detector, which applies feature-wise two-sample Cramér-von Mises (CVM) tests.
For two empirical distributions :math:`F(z)` and :math:`F_{ref}(z)`, the CVM test statistic is defined as

.. math::
   W = \sum_{z\in k} \left| F(z) - F_{ref}(z) \right|^2

where :math:`k` is the joint sample. The CVM test is an alternative to the `Kolmogorov-Smirnov`_ (K-S) two-sample test, which
uses the maximum distance between two empirical distributions :math:`F(z)` and :math:`F_{ref}(z)`. By using the full joint
sample, the CVM can exhibit greater power against shifts in higher moments, such as variance changes.

For multivariate data, the detector applies a separate CVM test to each feature, and the p-values obtained for each feature
are aggregated either via the `Bonferroni`_ or the `False Discovery Rate`_ (FDR) correction. The Bonferroni correction is more
conservative and controls for the probability of at least one false positive. The FDR correction on the other hand allows for
an expected fraction of false positives to occur. As with other univariate detectors such as the `Kolmogorov-Smirnov`_ detector,
for high-dimensional data, we typically want to reduce the dimensionality before computing the feature-wise univariate FET
tests and aggregating those via the chosen correction method.

.. autoclass:: daml.metrics.drift.CVMDrift
   :members:
   :inherited-members:

Kolmogorov-Smirnov
^^^^^^^^^^^^^^^^^^

The drift detector applies feature-wise two-sample `Kolmogorov-Smirnov`_ (K-S) tests. For multivariate data, the obtained
p-values for each feature are aggregated either via the `Bonferroni`_ or the `False Discovery Rate`_ (FDR) correction.
The Bonferroni correction is more conservative and controls for the probability of at least one false positive. The FDR
correction on the other hand allows for an expected fraction of false positives to occur.

.. autoclass:: daml.metrics.drift.KSDrift
   :members:
   :inherited-members:

Maximum Mean Discrepancy
^^^^^^^^^^^^^^^^^^^^^^^^

The `Maximum Mean Discrepancy`_ (MMD) detector is a kernel-based method for multivariate 2 sample testing. The MMD is
a distance-based measure between 2 distributions *p* and *q* based on the mean embeddings :math:`\mu_{p}` and :math:`\mu_{q}`
in a reproducing kernel Hilbert space :math:`F`:

.. math::
   MMD(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}

We can compute unbiased estimates of :math:`MMD^2` from the samples of the 2 distributions after applying the kernel trick.
We use by default a radial basis function kernel, but users are free to pass their own kernel of preference to the detector.
We obtain a :math:`p`-value via a permutation test on the values of :math:`MMD^2`.

.. autoclass:: daml.metrics.drift.MMDDrift
   :members:
   :inherited-members:

Classifier Uncertainty
^^^^^^^^^^^^^^^^^^^^^^

The classifier uncertainty drift detector aims to directly detect drift that is likely to effect the performance of a model
of interest. The approach is to test for change in the number of instances falling into regions of the input space on which
the model is uncertain in its predictions. For each instance in the reference set the detector obtains the model's prediction
and some associated notion of uncertainty. The same is done for the test set and if significant differences in uncertainty
are detected (via a `Kolmogorov-Smirnov`_ test) then drift is flagged. The detector's reference set should be disjoint from
the model's training set (on which the model's confidence may be higher).

.. autoclass:: daml.metrics.drift.UncertaintyDrift
   :members:
   :inherited-members:

GaussianRBF
^^^^^^^^^^^

The GaussianRBF class implements a Gaussian kernel, also known as a `radial basis function`_ (RBF) kernel. It is used
to construct a covariance matrix for gaussian processes and is the default kernel used in the MMD drift detection test.

.. autoclass:: daml.metrics.drift.GaussianRBF
   :members:

LastSeenUpdate
^^^^^^^^^^^^^^

.. autoclass:: daml.metrics.drift.LastSeenUpdate
   :members:
   :inherited-members:
   
ReservoirSamplingUpdate
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: daml.metrics.drift.ReservoirSamplingUpdate
   :members:
   :inherited-members: