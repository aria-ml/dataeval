# Metadata Parity

## What is metadata parity?

Metadata parity is a means for assessing fairness in Machine Learning by looking for independence between metadata factors and class labels in a dataset. This assessment helps a user understand sources of potential bias before a model gets trained on it and inadvertently learns spurious correlations. Metadata parity provides a report on how strongly each metadata factor is independent from class labels. In an ideal case with zero bias, the probability of observing a class label would be independent from observing a specific metadata factor.

## Why is metadata statistical independence important?

A model trained on a dataset must avoid learning unintended bias. A common way in which bias manifests is when class labels are not statistically independent from metadata attributes. For example, consider a scenario where a user wants to train a model to classify images as cats or as dogs. The user's training dataset has labeled pictures of cats and dogs. Suppose that, in this dataset, all dog pictures were taken in Washington, and all cat pictures were taken in Arizona. A model could learn this spurious correlation, and could classify an image as a cat or dog by inspecting the location information, rather than by inspecting features of cats and dogs. Thus, a picture of a cat taken in Washington could be misclassified as a dog. Early detection and mitigation of metadata bias is critical for training unbiased and reliable models.

## Why use parity over other statistical methods?

Metadata can be analyzed with simple methods, but their scalability is poor. Statistical independence between class labels and metadata factors can be difficult to detect at-scale when inspecting data points or when counting frequencies of factors. Especially when the distribution of class labels is highly unbalanced or when there are many different metadata factors, there is a need for a streamlined way to identify which metadata factors are cause for concern. A metadata parity analysis can obtain summery statistics for large-scale datasets with complex metadata.

Several other methods for assessing bias and fairness are commonly used, such as error rate balances, test-fairness, positive/negative class balance, and equal-confusion fairness. However, such metrics are calculated based on model predictions and probabilities, and as such they have to be evaluated after a model is already trained. Statistical parity of metadata can be computed before a model is trained, which enables faster iteration when developing unbiased ML pipelines.

## What can be done with the metadata parity information?

If all metadata factors are independent from labels, then the dataset is likely in good condition; a model trained on it will be less likely to overfit to spurious correlations.

If a metadata factor is not independent from class labels, then a model trained on the dataset could exhibit unintended bias. In this case, action is recommended. Actions include, but are not limited to:

1. Collecting or generating additional training data that has consistent label distributions across all values of the metadata factor.
2. Identifying how the spurious correlation manifests in the embeddings in a model, and subtracting out the bias in latent space.
3. Assigning weights to the loss function that de-emphasize samples that exhibit spurious correlations.

## See Also

- [Fairness In Machine Learning: A Survey](https://arxiv.org/abs/2010.04053)
- [Chi-Squared Test](https://en.wikipedia.org/wiki/Chi-squared_test)