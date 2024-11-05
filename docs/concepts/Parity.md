# Parity

## What is parity?

{term}`Parity` is a means for assessing fairness in {term}`machine learning<Machine Learning (ML)>` by looking for statistical independence between metadata factors and class labels in a dataset.
This assessment helps a user understand sources of potential {term}`bias<Bias>` before a model gets trained on it and inadvertently learns spurious correlations.
In an ideal case with zero bias, the probability of observing a class label would be independent from observing a specific metadata factor.

## Why is {term}`statistical independence<Statistical Independence>` important for metadata?

A model trained on a dataset must avoid learning unintended {term}`bias<Bias>`.
A common way in which bias manifests is when class labels are not statistically independent from metadata attributes.
For example, consider a scenario where a user wants to train a model to classify images as cats or as dogs.
Suppose that, in this dataset, all dog pictures were taken in Washington, and all cat pictures were taken in Arizona.
A model could learn this spurious correlation, and could classify an image as a cat or dog by inspecting the location information, 
rather than by inspecting features of cats and dogs.
Thus, a picture of a cat taken in Washington could be misclassified as a dog.
Early detection and mitigation of metadata bias is critical for training unbiased and reliable models.

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