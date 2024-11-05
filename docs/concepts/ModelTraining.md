# Model Training Methods

## What is it

## When to use it

The `AETrainer` class should be used when you have lots of images, have very large images, or your given speed requirements are strict.
Images can be hundreds of pixels tall and wide, and can have multiple channels, which makes them very large.
Encoding these images using an {term}`autoencoder<Autoencoder>` can shrink the images significantly.

## Theory behind it

### AE Outlier detection
The encoder is trained to create dense {term}`embeddings<Embeddings>` for the images while the decoder is trained
to reconstruct the new embedding into the original input image. The distances from the reconstructions
between the test images and original images, or the {term}`probability distribution<Probability Distribution>` differences are used to
measure how different they are and allow for the detection of Outliers.