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

**How it works:**

1. Label reference data as class 0 and analysis data as class 1
2. Train a binary classifier (LightGBM by default) using stratified k-fold
   {term}`cross-validation<Cross-Validation>`
3. Compute AUROC on held-out folds:

   $$
   \text{AUROC} = P(\hat{y}_{analysis} > \hat{y}_{ref})
   $$

   where $\hat{y}$ are predicted probabilities
4. Interpret AUROC values:
   - **AUROC ≈ 0.5**: No drift (classifier cannot distinguish datasets)
   - **AUROC > 0.65**: Significant drift detected (classifier can discriminate)
   - **AUROC < 0.45**: Potential data quality issues

**Key characteristics:**

- **Multivariate**: Captures drift across all features simultaneously
- **Model-based**: Leverages gradient boosting for complex pattern detection
- **Interpretable metric**: AUROC provides intuitive drift magnitude
- **Flexible**: Can detect any distributional changes the classifier can learn
- **Robust**: Cross-validation prevents overfitting to noise

**When to use:**

- **High-dimensional data** with complex feature interactions
- **Deep learning embeddings** (ResNet, CLIP, ViT features)
- When you need a single multivariate drift score
- Detecting subtle distributional shifts across many features
- When interpretability of AUROC metric is valuable
- Complementing univariate methods for comprehensive monitoring

**Limitations:**

- Computationally expensive (trains multiple models via cross-validation)
- Cannot identify which specific features drifted
- Requires sufficient samples for reliable cross-validation
- May miss drift that doesn't affect discriminative patterns
- Performance depends on classifier hyperparameters
