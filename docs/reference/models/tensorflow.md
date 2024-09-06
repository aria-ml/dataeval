(tensorflow-models-ref)=
# Tensorflow Models

## Models

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.AE(encoder_net: keras.Model, decoder_net: keras.Model)
```

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.AEGMM(encoder_net: keras.Model, decoder_net: keras.Model, gmm_density_net: keras.Model, n_gmm: int, recon_features: Callable = eucl_cosim_features)
```

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.PixelCNN(image_shape: tuple, conditional_shape: tuple | None = None, num_resnet: int = 5, num_hierarchies: int = 3, num_filters: int = 160, num_logistic_mix: int = 10, receptive_field_dims: tuple = (3, 3), dropout_p: float = 0.5, resnet_activation: str = "concat_elu", l2_weight: float = 0.0, use_weight_norm: bool = True, use_data_init: bool = True, high: int = 255, low: int = 0, dtype=tf.float32, name: str = "PixelCNN")
```

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.VAE(encoder_net: keras.Model, decoder_net: keras.Model, latent_dim: int, beta: float = 1.0)
```

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.VAEGMM(encoder_net: keras.Model, decoder_net: keras.Model, gmm_density_net: keras.Model, n_gmm: int, latent_dim: int, recon_features: Callable = eucl_cosim_features, beta: float = 1.0)
```

## Reconstruction Functions

```{eval-rst}
.. autofunction:: dataeval.models.tensorflow.eucl_cosim_features
```

## Loss Function Classes

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.LossGMM
```

```{eval-rst}
.. autoclass:: dataeval.models.tensorflow.Elbo
```

## Utility Functions

```{eval-rst}
.. autofunction:: dataeval.models.tensorflow.create_model
```
