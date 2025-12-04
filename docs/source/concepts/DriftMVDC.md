# _Multi-Variate Domain Classifier_

The {term}`Domain Classifier<Domain Classifier (DC)>` is a discriminative method
used for detecting multivariate drift by assessing how distinguishable reference
and analysis data distributions are from each other. The DC test statistic is
computed using a machine learning classifier (typically LightGBM) that attempts
to discriminate between two datasets:

$$
\textrm{DC} = \textrm{AUROC}(C(X_{\textrm{ref}}, X_{\textrm{analysis}}))
$$

where $C$ represents the classifier trained to distinguish between reference
data $X_{\textrm{ref}}$ and analysis data $X_{\textrm{analysis}}$, and $\textrm{AUROC}$ is the
{term}`area under the receiver operating characteristic curve<AUROC>`.

The Domain Classifier is particularly effective at detecting subtle shifts in
the joint distribution of features that may not be apparent when examining
individual features in isolation. When no drift is present, the AUROC score
approaches 0.5, indicating the classifier cannot effectively distinguish between
the datasets. As drift increases, the AUROC score rises toward 1.0, signifying
that the distributions have become increasingly distinguishable.

For implementation, the classifier undergoes
{term}`cross-validation<Cross-Validation>` to ensure robust discrimination
assessment. A threshold for significant change is typically defined based on the
variance of AUROC scores observed in a reference period, with alerts triggered
when scores exceed this threshold. The Domain Classifier method is particularly
suitable for complex, non-linear relationships in data and offers greater
sensitivity for detecting subtle multivariate shifts compared to
reconstruction-based approaches.
