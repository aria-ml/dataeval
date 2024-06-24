# Tensorflow Models

The Tensorflow models provided are tailored for usage with the outlier
detection metrics. DAML provides both basic default models through the
utility function <span class="title-ref">create_model</span> as well as
constructors which allow for customization of the encoder, decoder and
any other applicable layers used by the model.

**How does it work?**

The encoder is trained to create dense embeddings for the images while
the decoder is trained to reconstruct the new embedding into the
original input image. The distances from the reconstructions between the
test images and original images, or the probability distribution
differences are used to measure how different they are and allow for the
detection of outliers.

## Tutorials

There are no tutorials for Tensorflow models yet, but we will be adding
one soon.

## How To Guides

There are currently no how to's for Tensorflow models. If there are
scenarios that you want us to explain, contact us!

## DAML API

### Models

<div class="autoclass">

daml.models.tensorflow.AE(encoder_net: keras.Model, decoder_net:
keras.Model)

</div>

<div class="autoclass">

daml.models.tensorflow.AEGMM(encoder_net: keras.Model, decoder_net:
keras.Model, gmm_density_net: keras.Model, n_gmm: int, recon_features:
Callable = eucl_cosim_features)

</div>

<div class="autoclass">

daml.models.tensorflow.PixelCNN(image_shape: tuple, conditional_shape:
Optional\[tuple\] = None, num_resnet: int = 5, num_hierarchies: int = 3,
num_filters: int = 160, num_logistic_mix: int = 10,
receptive_field_dims: tuple = (3, 3), dropout_p: float = 0.5,
resnet_activation: str = "concat_elu", l2_weight: float = 0.0,
use_weight_norm: bool = True, use_data_init: bool = True, high: int =
255, low: int = 0, dtype=tf.float32, name: str = "PixelCNN")

</div>

<div class="autoclass">

daml.models.tensorflow.VAE(encoder_net: keras.Model, decoder_net:
keras.Model, latent_dim: int, beta: float = 1.0)

</div>

<div class="autoclass">

daml.models.tensorflow.VAEGMM(encoder_net: keras.Model, decoder_net:
keras.Model, gmm_density_net: keras.Model, n_gmm: int, latent_dim: int,
recon_features: Callable = eucl_cosim_features, beta: float = 1.0)

</div>

### Reconstruction Functions

<div class="autofunction">

daml.models.tensorflow.eucl_cosim_features

</div>

### Loss Function Classes

<div class="autoclass">

daml.models.tensorflow.LossGMM

</div>

<div class="autoclass">

daml.models.tensorflow.Elbo

</div>

### Utility Functions

<div class="autofunction">

daml.models.tensorflow.create_model

</div>
