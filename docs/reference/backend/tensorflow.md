(tensorflow-backend-ref)=
# Tensorflow Backend

## Models

```{eval-rst}
.. autoclass:: dataeval.tensorflow.models.AE(encoder_net: keras.Model, decoder_net: keras.Model)
.. autoclass:: dataeval.tensorflow.models.AEGMM(encoder_net: keras.Model, decoder_net: keras.Model, gmm_density_net: keras.Model, n_gmm: int, recon_features: Callable = eucl_cosim_features)
.. autoclass:: dataeval.tensorflow.models.PixelCNN(image_shape: tuple, conditional_shape: tuple | None = None, num_resnet: int = 5, num_hierarchies: int = 3, num_filters: int = 160, num_logistic_mix: int = 10, receptive_field_dims: tuple = (3, 3), dropout_p: float = 0.5, resnet_activation: str = "concat_elu", l2_weight: float = 0.0, use_weight_norm: bool = True, use_data_init: bool = True, high: int = 255, low: int = 0, dtype=tf.float32, name: str = "PixelCNN")
.. autoclass:: dataeval.tensorflow.models.VAE(encoder_net: keras.Model, decoder_net: keras.Model, latent_dim: int, beta: float = 1.0)
.. autoclass:: dataeval.tensorflow.models.VAEGMM(encoder_net: keras.Model, decoder_net: keras.Model, gmm_density_net: keras.Model, n_gmm: int, latent_dim: int, recon_features: Callable = eucl_cosim_features, beta: float = 1.0)
.. autofunction:: dataeval.tensorflow.models.create_model
```

## Reconstruction Functions

```{eval-rst}
.. autofunction:: dataeval.tensorflow.recon.eucl_cosim_features
```

## Loss Function Classes

```{eval-rst}
.. autoclass:: dataeval.tensorflow.loss.Elbo
.. autoclass:: dataeval.tensorflow.loss.LossGMM
```
