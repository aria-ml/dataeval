# Parity

## What is it

{term}`Parity` is a means for assessing fairness in {term}`machine learning<Machine Learning (ML)>` by looking for statistical independence between metadata factors and class labels in a dataset.
This assessment helps a user understand sources of potential {term}`bias<Bias>` before a model gets trained on it and inadvertently learns spurious correlations.
In an ideal case with zero bias, the probability of observing a class label would be independent from observing a specific metadata factor.

## When to use it

For both model and dataset development it is important to understand
correlational relationships that underlie the dataset.  Often, opportunities for
data collection are sparse, available only in non-operational locations and
conditions, with limited target diversity, etc.  A model trained on these
realistic datasets could learn to use secondary information to perform the
primary learning task, reducing the model's ability to generalize to new domains
or to perform unexpectedly when presented with new data.  Parity metric
provides a method for identifying *linear* relationships between dataset factors and class
labels _a priori_.  A T&E engineer or model developer should then use that information to
design tests for model generalization or data augmentation to mitigate the
opportunity for shortcut learning or sampling imbalance.

In order to use {term}`parity<Parity>`, the user must supply their metadata in a DataEval
specific format. Because of this requirement, DataEval has a `metadata_preprocessing` function
that will take in user [metadata](Metadata.md) and format it into DataEval's format. The parity function takes
in the output of the {func}`.metadata_preprocessing` function for its analysis.

## Why use parity over other statistical methods?

{term}`Parity` measures {term}`bias<Bias>` on the dataset prior to model testing allowing for faster iterations in developing unbiased ML pipelines.
Several methods such as error rate balances, test-fairness, positive/negative class {term}`balance<Balance>`, and equal-confusion fairness, 
which are commonly used for assessing bias and fairness, are calculated based on model predictions and probabilities.
Thus, those methods have to be evaluated after a model is already trained.

## What can be done with the parity information?

If all metadata factors are independent from labels, a model trained on it will be less likely to overfit to spurious correlations.

If a metadata factor is not independent from class labels, then a model trained on the dataset could exhibit unintended {term}`bias<Bias>`.
In this case, action is recommended. Actions include, but are not limited to:

1. Collecting or generating additional training data that has consistent label distributions across all values of the metadata factor.
2. Identifying how the spurious correlation manifests in the {term}`embeddings<Embeddings>` in a model, and subtracting out the bias in {term}`latent space<Latent Space>`.
3. Assigning weights to the loss function that de-emphasize samples that exhibit spurious correlations.

## See Also

- [Fairness In Machine Learning: A Survey](https://arxiv.org/abs/2010.04053)
- [Chi-Squared Test](https://en.wikipedia.org/wiki/Chi-squared_test)