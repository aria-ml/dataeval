# Detecting Out of Distribution Data

## What is it

`OOD_AE` is a method for detecting out-of-distribution images via autoencoder reconstruction error. Images which are poorly reconstructed by an autoencoder, trained on the reference dataset, are likely to be qualitatively different from those on which the model was trained. Much of the functionality comes from [Alibi Detect](https://github.com/SeldonIO/alibi-detect).
## When to use it

The `OOD_AE` class and similar should be used when you would like to find individual images in a dataset which are qualitatively different from those in a reference (training) dataset. Typically, the main use-case is when you have a new set of (operational) images, and would like to determine if there are any qualitatively different images amongst them. These could be a novel, operationally relevant class or sub-class which was not present in the training data. This type of detection is critical because models are likely to degrade rapidly if novel images represent a significant portion of operational data.

## Theory behind it

An autoencoder is a neural network which takes input data, compresses it down to a smaller dimensional space, and then attempts to reconstruct the original input data from the compressed data.
![ae](./images/ae.png) (https://www.compthree.com/blog/autoencoder/)

If a trained autoencoder encounters an image which falls outside the data manifold on which it is trained, it will generally do a poor job of reconstructing it. By default, we take the top percentile of reconstruction error from the training dataset, and set that as the threshold for considering an image as OOD.

Following OOD detection, a user can then investigate the individual images which were detected as OOD.

## References
[1] [Van Looveren, A., Klaise, J., Vacanti, G., Cobb, O., Scillitoe, A., Samoilescu, R., & Athorne, A. (2024). Alibi Detect: Algorithms for outlier, adversarial and drift detection (0.12.1.dev0)](https://github.com/SeldonIO/alibi-detect)