# type: ignore

"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, cast

import numpy as np

from dataeval.utils.lazy import lazyload

if TYPE_CHECKING:
    import tensorflow as tf
    import tensorflow_probability.python.bijectors as bijectors
    import tensorflow_probability.python.distributions as distributions
    import tensorflow_probability.python.internal as tfp_internal
    import tf_keras as keras
else:
    tf = lazyload("tensorflow")
    bijectors = lazyload("tensorflow_probability.python.bijectors")
    distributions = lazyload("tensorflow_probability.python.distributions")
    tfp_internal = lazyload("tensorflow_probability.python.internal")
    keras = lazyload("tf_keras")


def relative_euclidean_distance(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-12, axis: int = -1) -> tf.Tensor:
    """
    Relative Euclidean distance.

    Parameters
    ----------
    x
        Tensor used in distance computation.
    y
        Tensor used in distance computation.
    eps
        Epsilon added to denominator for numerical stability.
    axis
        Axis used to compute distance.

    Returns
    -------
    Tensor with relative Euclidean distance across specified axis.
    """
    denom = tf.concat(
        [
            tf.reshape(tf.norm(x, ord=2, axis=axis), (-1, 1)),  # type: ignore
            tf.reshape(tf.norm(y, ord=2, axis=axis), (-1, 1)),  # type: ignore
        ],
        axis=1,
    )
    dist = tf.norm(tf.math.subtract(x, y), ord=2, axis=axis) / (tf.reduce_min(denom, axis=axis) + eps)  # type: ignore
    return dist


def eucl_cosim_features(x: tf.Tensor, y: tf.Tensor, max_eucl: float = 1e2) -> tf.Tensor:
    """
    Compute features extracted from the reconstructed instance using the
    relative Euclidean distance and cosine similarity between 2 tensors.

    Parameters
    ----------
    x : tf.Tensor
        Tensor used in feature computation.
    y : tf.Tensor
        Tensor used in feature computation.
    max_eucl : float, default 1e2
        Maximum value to clip relative Euclidean distance by.

    Returns
    -------
    tf.Tensor
        Tensor concatenating the relative Euclidean distance and cosine similarity features.
    """
    if len(x.shape) > 2 or len(y.shape) > 2:
        x = cast(tf.Tensor, keras.layers.Flatten()(x))
        y = cast(tf.Tensor, keras.layers.Flatten()(y))
    rec_cos = tf.reshape(keras.losses.cosine_similarity(y, x, -1), (-1, 1))
    rec_euc = tf.reshape(relative_euclidean_distance(y, x, -1), (-1, 1))
    # rec_euc could become very large so should be clipped
    rec_euc = tf.clip_by_value(rec_euc, 0, max_eucl)
    return cast(tf.Tensor, tf.concat([rec_cos, rec_euc], -1))


class Sampling(keras.layers.Layer):
    """Reparametrization trick - Uses (z_mean, z_log_var) to sample the latent vector z."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Sample z.

        Parameters
        ----------
        inputs
            Tuple with mean and log :term:`variance<Variance>`.

        Returns
        -------
        Sampled vector z.
        """
        z_mean, z_log_var = inputs
        batch, dim = tuple(tf.shape(z_mean).numpy().ravel()[:2])  # type: ignore
        epsilon = cast(tf.Tensor, keras.backend.random_normal(shape=(batch, dim)))
        return z_mean + tf.exp(tf.math.multiply(0.5, z_log_var)) * epsilon


class EncoderAE(keras.layers.Layer):
    def __init__(self, encoder_net: keras.Sequential) -> None:
        """
        Encoder of AE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a keras.keras.Sequential class.
        name
            Name of encoder.
        """
        super().__init__(name="encoder_ae")
        self.encoder_net: keras.Sequential = encoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return cast(tf.Tensor, self.encoder_net(x))


class EncoderVAE(keras.layers.Layer):
    def __init__(self, encoder_net: keras.Sequential, latent_dim: int) -> None:
        """
        Encoder of VAE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a keras.keras.Sequential class.
        latent_dim
            Dimensionality of the :term:`latent space<Latent Space>`.
        name
            Name of encoder.
        """
        super().__init__(name="encoder_vae")
        self.encoder_net: keras.Sequential = encoder_net
        self._fc_mean = keras.layers.Dense(latent_dim, activation=None)
        self._fc_log_var = keras.layers.Dense(latent_dim, activation=None)
        self._sampling = Sampling()

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = cast(tf.Tensor, self.encoder_net(x))
        if len(x.shape) > 2:
            x = cast(tf.Tensor, keras.layers.Flatten()(x))
        z_mean = cast(tf.Tensor, self._fc_mean(x))
        z_log_var = cast(tf.Tensor, self._fc_log_var(x))
        z = cast(tf.Tensor, self._sampling((z_mean, z_log_var)))
        return z_mean, z_log_var, z


class Decoder(keras.layers.Layer):
    def __init__(self, decoder_net: keras.Sequential) -> None:
        """
        Decoder of AE and VAE.

        Parameters
        ----------
        decoder_net
            Layers for the decoder wrapped in a keras.keras.Sequential class.
        name
            Name of decoder.
        """
        super().__init__(name="decoder")
        self.decoder_net: keras.Sequential = decoder_net

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return cast(tf.Tensor, self.decoder_net(inputs))


class AE(keras.Model):
    """
    Combine encoder and decoder in AE.

    Parameters
    ----------
    encoder_net : keras.Sequential
        Layers for the encoder wrapped in a keras.keras.Sequential class.
    decoder_net : keras.Sequential
        Layers for the decoder wrapped in a keras.keras.Sequential class.
    """

    def __init__(self, encoder_net: keras.Sequential, decoder_net: keras.Sequential) -> None:
        super().__init__(name="ae")
        self.encoder: keras.layers.Layer = EncoderAE(encoder_net)
        self.decoder: keras.layers.Layer = Decoder(decoder_net)

    def call(self, inputs: tf.Tensor, training: bool | None = None, mask: tf.Tensor | None = None) -> tf.Tensor:
        z = cast(tf.Tensor, self.encoder(inputs))
        x_recon = cast(tf.Tensor, self.decoder(z))
        return x_recon


class VAE(keras.Model):
    """
    Combine encoder and decoder in VAE.

    Parameters
    ----------
    encoder_net : keras.Sequential
        Layers for the encoder wrapped in a keras.keras.Sequential class.
    decoder_net : keras.Sequential
        Layers for the decoder wrapped in a keras.keras.Sequential class.
    latent_dim : int
        Dimensionality of the :term:`latent space<Latent Space>`.
    beta : float, default 1.0
        Beta parameter for KL-divergence loss term.
    """

    def __init__(
        self, encoder_net: keras.Sequential, decoder_net: keras.Sequential, latent_dim: int, beta: float = 1.0
    ) -> None:
        super().__init__(name="vae_model")
        self.encoder: keras.layers.Layer = EncoderVAE(encoder_net, latent_dim)
        self.decoder: keras.layers.Layer = Decoder(decoder_net)
        self.beta: float = beta
        self.latent_dim: int = latent_dim

    def call(self, inputs: tf.Tensor, training: bool | None = None, mask: tf.Tensor | None = None) -> tf.Tensor:
        z_mean, z_log_var, z = cast(tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.encoder(inputs))
        x_recon = self.decoder(z)
        # add KL divergence loss term
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return cast(tf.Tensor, x_recon)


class AEGMM(keras.Model):
    """
    Deep Autoencoding Gaussian Mixture Model.

    Parameters
    ----------
    encoder_net : keras.Sequential
        Layers for the encoder wrapped in a keras.keras.Sequential class.
    decoder_net : keras.Sequential
        Layers for the decoder wrapped in a keras.keras.Sequential class.
    gmm_density_net : keras.Sequential
        Layers for the GMM network wrapped in a keras.keras.Sequential class.
    n_gmm : int
        Number of components in GMM.
    """

    def __init__(
        self,
        encoder_net: keras.Sequential,
        decoder_net: keras.Sequential,
        gmm_density_net: keras.Sequential,
        n_gmm: int,
    ) -> None:
        super().__init__("aegmm")
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.gmm_density = gmm_density_net
        self.n_gmm = n_gmm

    def call(
        self, inputs: tf.Tensor, training: bool | None = None, mask: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        enc = self.encoder(inputs)
        x_recon = cast(tf.Tensor, self.decoder(enc))
        recon_features = eucl_cosim_features(inputs, x_recon)
        z = cast(tf.Tensor, tf.concat([enc, recon_features], -1))
        gamma = cast(tf.Tensor, self.gmm_density(z))
        return x_recon, z, gamma


class VAEGMM(keras.Model):
    """
    Variational Autoencoding Gaussian Mixture Model.

    Parameters
    ----------
    encoder_net : keras.Sequential
        Layers for the encoder wrapped in a keras.keras.Sequential class.
    decoder_net : keras.Sequential
        Layers for the decoder wrapped in a keras.keras.Sequential class.
    gmm_density_net : keras.Sequential
        Layers for the GMM network wrapped in a keras.keras.Sequential class.
    n_gmm : int
        Number of components in GMM.
    latent_dim : int
        Dimensionality of the :term:`latent space<Latent Space>`.
    beta : float, default 1.0
        Beta parameter for KL-divergence loss term.
    """

    def __init__(
        self,
        encoder_net: keras.Sequential,
        decoder_net: keras.Sequential,
        gmm_density_net: keras.Sequential,
        n_gmm: int,
        latent_dim: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__(name="vaegmm")
        self.encoder: keras.Sequential = EncoderVAE(encoder_net, latent_dim)
        self.decoder: keras.Sequential = decoder_net
        self.gmm_density: keras.Sequential = gmm_density_net
        self.n_gmm: int = n_gmm
        self.latent_dim: int = latent_dim
        self.beta = beta

    def call(
        self, inputs: tf.Tensor, training: bool | None = None, mask: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        enc_mean, enc_log_var, enc = cast(tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.encoder(inputs))
        x_recon = cast(tf.Tensor, self.decoder(enc))
        recon_features = eucl_cosim_features(inputs, x_recon)
        z = cast(tf.Tensor, tf.concat([enc, recon_features], -1))
        gamma = cast(tf.Tensor, self.gmm_density(z))
        # add KL divergence loss term
        kl_loss = -0.5 * tf.reduce_mean(enc_log_var - tf.square(enc_mean) - tf.exp(enc_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return x_recon, z, gamma


class WeightNorm(keras.layers.Wrapper):
    def __init__(self, layer, data_init: bool = True, **kwargs) -> None:
        """Layer wrapper to decouple magnitude and direction of the layer's weights.

        This wrapper reparameterizes a layer by decoupling the weight's
        magnitude and direction. This speeds up convergence by improving the
        conditioning of the optimization problem. It has an optional data-dependent
        initialization scheme, in which initial values of weights are set as functions
        of the first minibatch of data. Both the weight normalization and data-
        dependent initialization are described in [Salimans and Kingma (2016)][1].

        Parameters
        ----------
        layer
            A `keras.layers.Layer` instance. Supported layer types are
            `Dense`, `Conv2D`, and `Conv2DTranspose`. Layers with multiple inputs
            are not supported.
        data_init
            If `True` use data dependent variable initialization.
        **kwargs
            Additional keyword args passed to `keras.layers.Wrapper`.

        Raises
        ------
        ValueError
            If `layer` is not a `keras.layers.Layer` instance.
        """
        if not isinstance(layer, keras.layers.Layer):
            raise ValueError(
                "Please initialize `WeightNorm` layer with a `keras.layers.Layer` " f"instance. You passed: {layer}"
            )

        layer_type = type(layer).__name__
        if layer_type not in ["Dense", "Conv2D", "Conv2DTranspose"]:
            warnings.warn(
                "`WeightNorm` is tested only for `Dense`, `Conv2D`, and "
                f"`Conv2DTranspose` layers. You passed a layer of type `{layer_type}`"
            )

        super().__init__(layer, **kwargs)

        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        self.filter_axis = -2 if layer_type == "Conv2DTranspose" else -1

    def _compute_weights(self):
        """Generate weights with normalization."""
        # Determine the axis along which to expand `g` so that `g` broadcasts to
        # the shape of `v`.
        new_axis = -self.filter_axis - 3

        self.layer.kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * tf.expand_dims(self.g, new_axis)

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=self.kernel_norm_axes))
        self.g.assign(kernel_norm)

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        # Normalize kernel first so that calling the layer calculates
        # `tf.dot(v, x)/tf.norm(v)` as in (5) in ([Salimans and Kingma, 2016][1]).
        self._compute_weights()

        activation = self.layer.activation
        self.layer.activation = None

        use_bias = self.layer.bias is not None
        if use_bias:
            bias = self.layer.bias
            self.layer.bias = tf.zeros_like(bias)

        # Since the bias is initialized as zero, setting the activation to zero and
        # calling the initialized layer (with normalized kernel) yields the correct
        # computation ((5) in Salimans and Kingma (2016))
        x_init = self.layer(inputs)
        norm_axes_out = list(range(x_init.shape.rank - 1))
        m_init, v_init = tf.nn.moments(x_init, norm_axes_out)
        scale_init = 1.0 / tf.sqrt(v_init + 1e-10)

        self.g.assign(self.g * scale_init)
        if use_bias:
            self.layer.bias = bias
            self.layer.bias.assign(-m_init * scale_init)
        self.layer.activation = activation

    def build(self, input_shape=None):
        """Build `Layer`.

        Parameters
        ----------
        input_shape
            The shape of the input to `self.layer`.

        Raises
        ------
        ValueError
            If `Layer` does not contain a `kernel` of weights.
        """
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[0] = None
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, "kernel"):
                raise ValueError("`WeightNorm` must wrap a layer that contains a `kernel` for weights")

            self.kernel_norm_axes = list(range(self.layer.kernel.shape.ndims))
            self.kernel_norm_axes.pop(self.filter_axis)

            self.v = self.layer.kernel

            # to avoid a duplicate `kernel` variable after `build` is called
            self.layer.kernel = None
            self.g = self.add_weight(
                name="g",
                shape=(int(self.v.shape[self.filter_axis]),),
                initializer="ones",
                dtype=self.v.dtype,
                trainable=True,
            )
            self.initialized = self.add_weight(name="initialized", dtype=tf.bool, trainable=False)
            self.initialized.assign(False)

        super().build()

    @tf.function
    def call(self, inputs):
        """Call `Layer`."""
        if not self.initialized:
            if self.data_init:
                self._data_dep_init(inputs)
            else:  # initialize `g` as the norm of the initialized kernel
                self._init_norm()

            self.initialized.assign(True)

        self._compute_weights()
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())


class Shift(bijectors.Bijector):
    def __init__(self, shift, validate_args=False, name="shift") -> None:
        """Instantiates the `Shift` bijector which computes `Y = g(X; shift) = X + shift`
        where `shift` is a numeric `Tensor`.

        Parameters
        ----------
        shift
            Floating-point `Tensor`.
        validate_args
            Python `bool` indicating whether arguments should be checked for correctness.
        name
            Python `str` name given to ops managed by this object.
        """
        with tf.name_scope(name) as name:
            dtype = tfp_internal.dtype_util.common_dtype([shift], dtype_hint=tf.float32)
            self._shift = tfp_internal.tensor_util.convert_nonref_to_tensor(shift, dtype=dtype, name="shift")
            super().__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=True,
                dtype=dtype,
                validate_args=validate_args,
                name=name,
            )

    @property
    def shift(self):
        """The `shift` `Tensor` in `Y = X + shift`."""
        return self._shift

    @classmethod
    def _is_increasing(cls):
        return True

    def _forward(self, x):
        return x + self.shift

    def _inverse(self, y):
        return y - self.shift

    def _forward_log_det_jacobian(self, x):
        # is_constant_jacobian = True for this bijector, hence the
        # `log_det_jacobian` need only be specified for a single input, as this will
        # be tiled to match `event_ndims`.
        return tf.zeros([], dtype=tfp_internal.dtype_util.base_dtype(x.dtype))


class PixelCNN(distributions.distribution.Distribution):
    """
    Construct Pixel CNN++ distributions.distribution.

    Parameters
    ----------
    image_shape : tuple
        3D `TensorShape` or tuple for the `[height, width, channels]` dimensions of the image.
    conditional_shape : tuple, optional - default None
        `TensorShape` or tuple for the shape of the conditional input, or `None` if there is no conditional input.
    num_resnet : int, default 5
        The number of layers (shown in Figure 2 of [2]) within each highest-level block of Figure 2 of [1].
    num_hierarchies : int, default 3
        The number of highest-level blocks (separated by expansions/contractions of dimensions in Figure 2 of [1].)
    num_filters : int, default 160
        The number of convolutional filters.
    num_logistic_mix : int, default 10
        Number of components in the distributions.logistic mixture distributions.distribution.
    receptive_field_dims tuple, default (3, 3)
        Height and width in pixels of the receptive field of the convolutional layers above and to the left
        of a given pixel. The width (second element of the tuple) should be odd. Figure 1 (middle) of [2]
        shows a receptive field of (3, 5) (the row containing the current pixel is included in the height).
        The default of (3, 3) was used to produce the results in [1].
    dropout_p : float, default 0.0
        The dropout probability. Should be between 0 and 1.
    resnet_activation : str, default "concat_elu"
        The type of activation to use in the resnet blocks. May be 'concat_elu', 'elu', or 'relu'.
    l2_weight : float, default 0.0
        The L2 regularization weight.
    use_weight_norm : bool, default True
        If `True` then use weight normalization (works only in Eager mode).
    use_data_init : bool, default True
        If `True` then use data-dependent initialization (has no effect if `use_weight_norm` is `False`).
    high : int, default 255
        The maximum value of the input data (255 for an 8-bit image).
    low : int, default 0
        The minimum value of the input data.
    dtype : tensorflow dtype, default tf.float32
        Data type of the `Distribution`.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        conditional_shape: tuple[int, ...] | None = None,
        num_resnet: int = 5,
        num_hierarchies: int = 3,
        num_filters: int = 160,
        num_logistic_mix: int = 10,
        receptive_field_dims: tuple[int, int] = (3, 3),
        dropout_p: float = 0.5,
        resnet_activation: str = "concat_elu",
        l2_weight: float = 0.0,
        use_weight_norm: bool = True,
        use_data_init: bool = True,
        high: int = 255,
        low: int = 0,
        dtype: tf.DType = tf.float32,
    ) -> None:
        parameters = dict(locals())
        with tf.name_scope("PixelCNN") as name:
            super().__init__(
                dtype=dtype,
                reparameterization_type=tfp_internal.reparameterization.NOT_REPARAMETERIZED,
                validate_args=False,
                allow_nan_stats=True,
                parameters=parameters,
                name=name,
            )

            if not tfp_internal.tensorshape_util.is_fully_defined(image_shape):
                raise ValueError("`image_shape` must be fully defined.")

            if conditional_shape is not None and not tfp_internal.tensorshape_util.is_fully_defined(conditional_shape):
                raise ValueError("`conditional_shape` must be fully defined.")

            if tfp_internal.tensorshape_util.rank(image_shape) != 3:
                raise ValueError("`image_shape` must have length 3, representing [height, width, channels] dimensions.")

            self._high = tf.cast(high, self.dtype)
            self._low = tf.cast(low, self.dtype)
            self._num_logistic_mix = num_logistic_mix
            self._network = PixelCNNNetwork(
                dropout_p=dropout_p,
                num_resnet=num_resnet,
                num_hierarchies=num_hierarchies,
                num_filters=num_filters,
                num_logistic_mix=num_logistic_mix,
                receptive_field_dims=receptive_field_dims,
                resnet_activation=resnet_activation,
                l2_weight=l2_weight,
                use_weight_norm=use_weight_norm,
                use_data_init=use_data_init,
                dtype=dtype,
            )

            image_input_shape = tfp_internal.tensorshape_util.concatenate([None], image_shape)
            if conditional_shape is None:
                input_shape = image_input_shape
            else:
                conditional_input_shape = tfp_internal.tensorshape_util.concatenate([None], conditional_shape)
                input_shape = [image_input_shape, conditional_input_shape]

            self.image_shape = image_shape
            self.conditional_shape = conditional_shape
            self._network.build(input_shape)

    def _make_mixture_dist(self, component_logits, locs, scales, return_per_feature: bool = False):
        """Builds a mixture of quantized distributions.logistic distributions.

        Parameters
        ----------
        component_logits
            4D `Tensor` of logits for the Categorical distributions.distribution
            over Quantized Logistic mixture components. Dimensions are `[batch_size,
            height, width, num_logistic_mix]`.
        locs
            4D `Tensor` of location parameters for the Quantized Logistic
            mixture components. Dimensions are `[batch_size, height, width,
            num_logistic_mix, num_channels]`.
        scales
            4D `Tensor` of location parameters for the Quantized Logistic
            mixture components. Dimensions are `[batch_size, height, width,
            num_logistic_mix, num_channels]`.
        return_per_feature
            If True, return per pixel level log prob.

        Returns
        -------
        dist
            A quantized distributions.logistic mixture `tfp.distributions.distribution` over the input data.
        """
        mixture_distribution = distributions.categorical.Categorical(logits=component_logits)

        # Convert distributions.distribution parameters for pixel values in
        # `[self._low, self._high]` for use with `QuantizedDistribution`
        locs = self._low + 0.5 * (self._high - self._low) * (locs + 1.0)
        scales *= 0.5 * (self._high - self._low)
        logistic_dist = distributions.quantized_distribution.QuantizedDistribution(
            distribution=distributions.transformed_distribution.TransformedDistribution(
                distribution=distributions.logistic.Logistic(loc=locs, scale=scales),
                bijector=Shift(shift=tf.cast(-0.5, self.dtype)),
            ),
            low=self._low,
            high=self._high,
        )

        # mixture with logistics for the loc and scale on each pixel for each component
        dist = distributions.mixture_same_family.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=distributions.independent.Independent(logistic_dist, reinterpreted_batch_ndims=1),
        )
        if return_per_feature:
            return dist
        else:
            return distributions.independent.Independent(dist, reinterpreted_batch_ndims=2)

    def _log_prob(self, value, conditional_input=None, training=None, return_per_feature=False):
        """Log probability function with optional conditional input.

        Calculates the log probability of a batch of data under the modeled
        distributions.distribution (or conditional distributions.distribution, if conditional input is
        provided).

        Parameters
        ----------
        value
            `Tensor` or :term:`NumPy` array of image data. May have leading batch
            dimension(s), which must broadcast to the leading batch dimensions of
            `conditional_input`.
        conditional_input
            `Tensor` on which to condition the distributions.distribution (e.g.
            class labels), or `None`. May have leading batch dimension(s), which
            must broadcast to the leading batch dimensions of `value`.
        training
            `bool` or `None`. If `bool`, it controls the dropout layer,
            where `True` implies dropout is active. If `None`, it defaults to
            `keras.backend.learning_phase()`.
        return_per_feature
            `bool`. If True, return per pixel level log prob.

        Returns
        -------
        log_prob_values: `Tensor`.
        """
        # Determine the batch shape of the input images
        image_batch_shape = tfp_internal.prefer_static.shape(value)[:-3]

        # Broadcast `value` and `conditional_input` to the same batch_shape
        if conditional_input is None:
            image_batch_and_conditional_shape = image_batch_shape
        else:
            conditional_input = tf.convert_to_tensor(conditional_input)
            conditional_input_shape = tfp_internal.prefer_static.shape(conditional_input)
            conditional_batch_rank = tfp_internal.prefer_static.rank(
                conditional_input
            ) - tfp_internal.tensorshape_util.rank(self.conditional_shape)
            conditional_batch_shape = conditional_input_shape[:conditional_batch_rank]

            image_batch_and_conditional_shape = tfp_internal.prefer_static.broadcast_shape(
                image_batch_shape, conditional_batch_shape
            )
            conditional_input = tf.broadcast_to(
                conditional_input,
                tfp_internal.prefer_static.concat([image_batch_and_conditional_shape, self.conditional_shape], axis=0),
            )
            value = tf.broadcast_to(
                value,
                tfp_internal.prefer_static.concat([image_batch_and_conditional_shape, self.event_shape], axis=0),
            )

            # Flatten batch dimension for input to Keras model
            conditional_input = tf.reshape(
                conditional_input,
                tfp_internal.prefer_static.concat([(-1,), self.conditional_shape], axis=0),
            )

        value = tf.reshape(value, tfp_internal.prefer_static.concat([(-1,), self.event_shape], axis=0))

        transformed_value = (2.0 * (value - self._low) / (self._high - self._low)) - 1.0
        inputs = transformed_value if conditional_input is None else [transformed_value, conditional_input]

        params = self._network(inputs, training=training)

        num_channels = self.event_shape[-1]
        if num_channels == 1:
            component_logits, locs, scales = params
        else:
            # If there is more than one channel, we create a linear autoregressive
            # dependency among the location parameters of the channels of a single
            # pixel (the scale parameters within a pixel are distributions.independent). For a pixel
            # with R/G/B channels, the `r`, `g`, and `b` saturation values are
            # distributed as:
            #
            # r ~ Logistic(loc_r, scale_r)
            # g ~ Logistic(coef_rg * r + loc_g, scale_g)
            # b ~ Logistic(coef_rb * r + coef_gb * g + loc_b, scale_b)
            # on the coefficients instead of split/multiply/concat
            component_logits, locs, scales, coeffs = params
            num_coeffs = num_channels * (num_channels - 1) // 2
            loc_tensors = tf.split(locs, num_channels, axis=-1)
            coef_tensors = tf.split(coeffs, num_coeffs, axis=-1)
            channel_tensors = tf.split(value, num_channels, axis=-1)

            coef_count = 0
            for i in range(num_channels):
                channel_tensors[i] = channel_tensors[i][..., tf.newaxis, :]
                for j in range(i):
                    loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
                    coef_count += 1
            locs = tf.concat(loc_tensors, axis=-1)

        dist = self._make_mixture_dist(component_logits, locs, scales, return_per_feature=return_per_feature)
        log_px = dist.log_prob(value)
        if return_per_feature:
            return log_px
        else:
            return tf.reshape(log_px, image_batch_and_conditional_shape)

    def _sample_n(self, n, seed=None, conditional_input=None, training=False):
        """Samples from the distributions.distribution, with optional conditional input.

        Parameters
        ----------
        n
            `int`, number of samples desired.
        seed
            `int`, seed for RNG. Setting a random seed enforces reproducibility
            of the samples between sessions (not within a single session).
        conditional_input
            `Tensor` on which to condition the distributions.distribution (e.g.
            class labels), or `None`.
        training
            `bool` or `None`. If `bool`, it controls the dropout layer,
            where `True` implies dropout is active. If `None`, it defers to Keras'
            handling of train/eval status.

        Returns
        -------
        samples
            a `Tensor` of shape `[n, height, width, num_channels]`.
        """
        if conditional_input is not None:
            conditional_input = tf.convert_to_tensor(conditional_input, dtype=self.dtype)
            conditional_event_rank = tfp_internal.tensorshape_util.rank(self.conditional_shape)
            conditional_input_shape = tfp_internal.prefer_static.shape(conditional_input)
            conditional_sample_rank = tfp_internal.prefer_static.rank(conditional_input) - conditional_event_rank

            # If `conditional_input` has no sample dimensions, prepend a sample
            # dimension
            if conditional_sample_rank == 0:
                conditional_input = conditional_input[tf.newaxis, ...]
                conditional_sample_rank = 1

            # Assert that the conditional event shape in the `PixelCnnNetwork` is the
            # same as that implied by `conditional_input`.
            conditional_event_shape = conditional_input_shape[conditional_sample_rank:]
            with tf.control_dependencies([tf.assert_equal(self.conditional_shape, conditional_event_shape)]):
                conditional_sample_shape = conditional_input_shape[:conditional_sample_rank]
                repeat = n // tfp_internal.prefer_static.reduce_prod(conditional_sample_shape)
                h = tf.reshape(
                    conditional_input,
                    tfp_internal.prefer_static.concat([(-1,), self.conditional_shape], axis=0),
                )
                h = tf.tile(
                    h,
                    tfp_internal.prefer_static.pad(
                        [repeat],
                        paddings=[[0, conditional_event_rank]],
                        constant_values=1,
                    ),
                )

        samples_0 = tf.random.uniform(
            tfp_internal.prefer_static.concat([(n,), self.event_shape], axis=0),
            minval=-1.0,
            maxval=1.0,
            dtype=self.dtype,
            seed=seed,
        )
        inputs = samples_0 if conditional_input is None else [samples_0, h]
        params_0 = self._network(inputs, training=training)
        samples_0 = self._sample_channels(*params_0, seed=seed)

        image_height, image_width, _ = tfp_internal.tensorshape_util.as_list(self.event_shape)

        def loop_body(index, samples):
            """Loop for iterative pixel sampling.

            Parameters
            ----------
            index
                0D `Tensor` of type `int32`. Index of the current pixel.
            samples
              4D `Tensor`. Images with pixels sampled in raster order, up to
              pixel `[index]`, with dimensions `[batch_size, height, width,
              num_channels]`.

            Returns
            -------
            samples
                4D `Tensor`. Images with pixels sampled in raster order, up to \
                and including pixel `[index]`, with dimensions `[batch_size, height, \
                width, num_channels]`.
            """
            inputs = samples if conditional_input is None else [samples, h]
            params = self._network(inputs, training=training)
            samples_new = self._sample_channels(*params, seed=seed)

            # Update the current pixel
            samples = tf.transpose(samples, [1, 2, 3, 0])
            samples_new = tf.transpose(samples_new, [1, 2, 3, 0])
            row, col = index // image_width, index % image_width
            updates = samples_new[row, col, ...][tf.newaxis, ...]
            samples = tf.tensor_scatter_nd_update(samples, [[row, col]], updates)
            samples = tf.transpose(samples, [3, 0, 1, 2])

            return index + 1, samples

        index0 = tf.zeros([], dtype=tf.int32)

        # Construct the while loop for sampling
        total_pixels = image_height * image_width
        loop_cond = lambda ind, _: tf.less(ind, total_pixels)  # noqa: E731
        init_vars = (index0, samples_0)
        _, samples = tf.while_loop(loop_cond, loop_body, init_vars, parallel_iterations=1)

        transformed_samples = self._low + 0.5 * (self._high - self._low) * (samples + 1.0)
        return tf.round(transformed_samples)

    def _sample_channels(self, component_logits, locs, scales, coeffs=None, seed=None):
        """Sample a single pixel-iteration and apply channel conditioning.

        Parameters
        ----------
        component_logits
            4D `Tensor` of logits for the Categorical distributions.distribution
            over Quantized Logistic mixture components. Dimensions are `[batch_size,
            height, width, num_logistic_mix]`.
        locs
            4D `Tensor` of location parameters for the Quantized Logistic
            mixture components. Dimensions are `[batch_size, height, width,
            num_logistic_mix, num_channels]`.
        scales
            4D `Tensor` of location parameters for the Quantized Logistic
            mixture components. Dimensions are `[batch_size, height, width,
            num_logistic_mix, num_channels]`.
        coeffs
            4D `Tensor` of coefficients for the linear dependence among color
            channels, or `None` if there is only one channel. Dimensions are
            `[batch_size, height, width, num_logistic_mix, num_coeffs]`, where
            `num_coeffs = num_channels * (num_channels - 1) // 2`.
        seed
            `int`, random seed.

        Returns
        -------
        samples
            4D `Tensor` of sampled image data with autoregression among \
            channels. Dimensions are `[batch_size, height, width, num_channels]`.
        """
        num_channels = self.event_shape[-1]

        # sample mixture components once for the entire pixel
        component_dist = distributions.categorical.Categorical(logits=component_logits)
        mask = tf.one_hot(indices=component_dist.sample(seed=seed), depth=self._num_logistic_mix)
        mask = tf.cast(mask[..., tf.newaxis], self.dtype)

        # apply mixture component mask and separate out RGB parameters
        masked_locs = tf.reduce_sum(locs * mask, axis=-2)
        loc_tensors = tf.split(masked_locs, num_channels, axis=-1)
        masked_scales = tf.reduce_sum(scales * mask, axis=-2)
        scale_tensors = tf.split(masked_scales, num_channels, axis=-1)

        if coeffs is not None:
            num_coeffs = num_channels * (num_channels - 1) // 2
            masked_coeffs = tf.reduce_sum(coeffs * mask, axis=-2)
            coef_tensors = tf.split(masked_coeffs, num_coeffs, axis=-1)

        channel_samples = []
        coef_count = 0
        for i in range(num_channels):
            loc = loc_tensors[i]
            for c in channel_samples:
                loc += c * coef_tensors[coef_count]
                coef_count += 1

            logistic_samp = distributions.logistic.Logistic(loc=loc, scale=scale_tensors[i]).sample(seed=seed)
            logistic_samp = tf.clip_by_value(logistic_samp, -1.0, 1.0)
            channel_samples.append(logistic_samp)

        return tf.concat(channel_samples, axis=-1)

    def _batch_shape(self):
        return tf.TensorShape([])

    def _event_shape(self):
        return tf.TensorShape(self.image_shape)


class PixelCNNNetwork(keras.layers.Layer):
    """Keras `Layer` to parameterize a Pixel CNN++ distributions.distribution.
    This is a Keras implementation of the Pixel CNN++ network, as described in
    Salimans et al. (2017)[1] and van den Oord et al. (2016)[2].
    (https://github.com/openai/pixel-cnn).
    #### References
    [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
       Likelihood and Other Modifications. In _International Conference on
       Learning Representations_, 2017.
       https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf
       Additional details at https://github.com/openai/pixel-cnn
    [2]: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt,
       Alex Graves, and Koray Kavukcuoglu. Conditional Image Generation with
       PixelCNN Decoders. In _30th Conference on Neural Information Processing
       Systems_, 2016.
       https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf.
    """

    def __init__(
        self,
        dropout_p: float = 0.5,
        num_resnet: int = 5,
        num_hierarchies: int = 3,
        num_filters: int = 160,
        num_logistic_mix: int = 10,
        receptive_field_dims: tuple[int, int] = (3, 3),
        resnet_activation: str = "concat_elu",
        l2_weight: float = 0.0,
        use_weight_norm: bool = True,
        use_data_init: bool = True,
        dtype: tf.DType = tf.float32,
    ) -> None:
        """Initialize the :term:`neural network<Neural Network>` for the Pixel CNN++ distributions.distribution.

        Parameters
        ----------
        dropout_p
            `float`, the dropout probability. Should be between 0 and 1.
        num_resnet
            `int`, the number of layers (shown in Figure 2 of [2]) within
            each highest-level block of Figure 2 of [1].
        num_hierarchies
            `int`, the number of hightest-level blocks (separated by
            expansions/contractions of dimensions in Figure 2 of [1].)
        num_filters
            `int`, the number of convolutional filters.
        num_logistic_mix
            `int`, number of components in the distributions.logistic mixture
            distributions.distribution.
        receptive_field_dims
            `tuple`, height and width in pixels of the receptive
            field of the convolutional layers above and to the left of a given
            pixel. The width (second element of the tuple) should be odd. Figure 1
            (middle) of [2] shows a receptive field of (3, 5) (the row containing
            the current pixel is included in the height). The default of (3, 3) was
            used to produce the results in [1].
        resnet_activation
            `string`, the type of activation to use in the resnet
            blocks. May be 'concat_elu', 'elu', or 'relu'.
        l2_weight
            `float`, the L2 regularization weight.
        use_weight_norm
            `bool`, if `True` then use weight normalization.
        use_data_init
            `bool`, if `True` then use data-dependent initialization
            (has no effect if `use_weight_norm` is `False`).
        dtype
            Data type of the layer.
        """
        super().__init__(dtype=dtype)
        self._dropout_p = dropout_p
        self._num_resnet = num_resnet
        self._num_hierarchies = num_hierarchies
        self._num_filters = num_filters
        self._num_logistic_mix = num_logistic_mix
        self._receptive_field_dims = receptive_field_dims  # first set desired receptive field, then infer kernel
        self._resnet_activation = resnet_activation
        self._l2_weight = l2_weight

        if use_weight_norm:

            def layer_wrapper(layer):
                def wrapped_layer(*args, **kwargs):
                    return WeightNorm(layer(*args, **kwargs), data_init=use_data_init)

                return wrapped_layer

            self._layer_wrapper = layer_wrapper
        else:
            self._layer_wrapper = lambda layer: layer

    def build(self, input_shape: tuple[int, ...]) -> None:
        dtype = self.dtype
        if len(input_shape) == 2:
            batch_image_shape, batch_conditional_shape = input_shape
            conditional_input = keras.layers.Input(shape=batch_conditional_shape[1:], dtype=dtype)
        else:
            batch_image_shape = input_shape
            conditional_input = None

        image_shape = batch_image_shape[1:]
        image_input = keras.layers.Input(shape=image_shape, dtype=dtype)

        if self._resnet_activation == "concat_elu":
            activation = keras.layers.Lambda(lambda x: tf.nn.elu(tf.concat([x, -x], axis=-1)), dtype=dtype)
        else:
            activation = keras.activations.get(self._resnet_activation)

        # Define layers with default inputs and layer wrapper applied
        Conv2D = functools.partial(  # pylint:disable=invalid-name
            self._layer_wrapper(keras.layers.Convolution2D),
            filters=self._num_filters,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(self._l2_weight),
            dtype=dtype,
        )

        Dense = functools.partial(  # pylint:disable=invalid-name
            self._layer_wrapper(keras.layers.Dense),
            kernel_regularizer=keras.regularizers.l2(self._l2_weight),
            dtype=dtype,
        )

        Conv2DTranspose = functools.partial(  # pylint:disable=invalid-name
            self._layer_wrapper(keras.layers.Conv2DTranspose),
            filters=self._num_filters,
            padding="same",
            strides=(2, 2),
            kernel_regularizer=keras.regularizers.l2(self._l2_weight),
            dtype=dtype,
        )

        rows, cols = self._receptive_field_dims

        # Define the dimensions of the valid (unmasked) areas of the layer kernels
        # for stride 1 convolutions in the internal layers.
        kernel_valid_dims = {
            "vertical": (rows - 1, cols),  # vertical stack
            "horizontal": (2, cols // 2 + 1),
        }  # horizontal stack

        # Define the size of the kernel necessary to center the current pixel
        # correctly for stride 1 convolutions in the internal layers.
        kernel_sizes = {"vertical": (2 * rows - 3, cols), "horizontal": (3, cols)}

        # Make the kernel constraint functions for stride 1 convolutions in internal
        # layers.
        kernel_constraints = {
            k: _make_kernel_constraint(kernel_sizes[k], (0, v[0]), (0, v[1])) for k, v in kernel_valid_dims.items()
        }

        # Build the initial vertical stack/horizontal stack convolutional layers,
        # as shown in Figure 1 of [2]. The receptive field of the initial vertical
        # stack layer is a rectangular area centered above the current pixel.
        vertical_stack_init = Conv2D(
            kernel_size=(2 * rows - 1, cols),
            kernel_constraint=_make_kernel_constraint((2 * rows - 1, cols), (0, rows - 1), (0, cols)),
        )(image_input)

        # In Figure 1 [2], the receptive field of the horizontal stack is
        # illustrated as the pixels in the same row and to the left of the current
        # pixel. [1] increases the height of this receptive field from one pixel to
        # two (`horizontal_stack_left`) and additionally includes a subset of the
        # row of pixels centered above the current pixel (`horizontal_stack_up`).
        horizontal_stack_up = Conv2D(
            kernel_size=(3, cols),
            kernel_constraint=_make_kernel_constraint((3, cols), (0, 1), (0, cols)),
        )(image_input)

        horizontal_stack_left = Conv2D(
            kernel_size=(3, cols),
            kernel_constraint=_make_kernel_constraint((3, cols), (0, 2), (0, cols // 2)),
        )(image_input)

        horizontal_stack_init = keras.layers.add([horizontal_stack_up, horizontal_stack_left], dtype=dtype)

        layer_stacks = {
            "vertical": [vertical_stack_init],
            "horizontal": [horizontal_stack_init],
        }

        # Build the downward pass of the U-net (left-hand half of Figure 2 of [1]).
        # Each `i` iteration builds one of the highest-level blocks (identified as
        # 'Sequence of 6 layers' in the figure, consisting of `num_resnet=5` stride-
        # 1 layers, and one stride-2 layer that contracts the height/width
        # dimensions). The `_` iterations build the stride 1 layers. The layers of
        # the downward pass are stored in lists, since we'll later need them to make
        # skip-connections to layers in the upward pass of the U-net (the skip-
        # connections are represented by curved lines in Figure 2 [1]).
        for i in range(self._num_hierarchies):
            for _ in range(self._num_resnet):
                # Build a layer shown in Figure 2 of [2]. The 'vertical' iteration
                # builds the layers in the left half of the figure, and the 'horizontal'
                # iteration builds the layers in the right half.
                for stack in ["vertical", "horizontal"]:
                    input_x = layer_stacks[stack][-1]
                    x = activation(input_x)
                    x = Conv2D(
                        kernel_size=kernel_sizes[stack],
                        kernel_constraint=kernel_constraints[stack],
                    )(x)

                    # Add the vertical-stack layer to the horizontal-stack layer
                    if stack == "horizontal":
                        h = activation(layer_stacks["vertical"][-1])
                        h = Dense(self._num_filters)(h)
                        x = keras.layers.add([h, x], dtype=dtype)

                    x = activation(x)
                    x = keras.layers.Dropout(self._dropout_p, dtype=dtype)(x)
                    x = Conv2D(
                        filters=2 * self._num_filters,
                        kernel_size=kernel_sizes[stack],
                        kernel_constraint=kernel_constraints[stack],
                    )(x)

                    if conditional_input is not None:
                        h_projection = _build_and_apply_h_projection(conditional_input, self._num_filters, dtype=dtype)
                        x = keras.layers.add([x, h_projection], dtype=dtype)

                    x = _apply_sigmoid_gating(x)

                    # Add a residual connection from the layer's input.
                    out = keras.layers.add([input_x, x], dtype=dtype)
                    layer_stacks[stack].append(out)

            if i < self._num_hierarchies - 1:
                # Build convolutional layers that contract the height/width dimensions
                # on the downward pass between each set of layers (e.g. contracting from
                # 32x32 to 16x16 in Figure 2 of [1]).
                for stack in ["vertical", "horizontal"]:
                    # Define kernel dimensions/masking to maintain the autoregressive property.
                    x = layer_stacks[stack][-1]
                    h, w = kernel_valid_dims[stack]
                    kernel_height = 2 * h
                    kernel_width = w + 1 if stack == "vertical" else 2 * w
                    kernel_size = (kernel_height, kernel_width)
                    kernel_constraint = _make_kernel_constraint(kernel_size, (0, h), (0, w))
                    x = Conv2D(
                        strides=(2, 2),
                        kernel_size=kernel_size,
                        kernel_constraint=kernel_constraint,
                    )(x)
                    layer_stacks[stack].append(x)

        # Upward pass of the U-net (right-hand half of Figure 2 of [1]). We stored
        # the layers of the downward pass in a list, in order to access them to make
        # skip-connections to the upward pass. For the upward pass, we need to keep
        # track of only the current layer, so we maintain a reference to the
        # current layer of the horizontal/vertical stack in the `upward_pass` dict.
        # The upward pass begins with the last layer of the downward pass.
        upward_pass = {key: stack.pop() for key, stack in layer_stacks.items()}

        # As with the downward pass, each `i` iteration builds a highest level block
        # in Figure 2 [1], and the `_` iterations build individual layers within the
        # block.
        for i in range(self._num_hierarchies):
            num_resnet = self._num_resnet if i == 0 else self._num_resnet + 1

            for _ in range(num_resnet):
                # Build a layer as shown in Figure 2 of [2], with a skip-connection
                # from the symmetric layer in the downward pass.
                for stack in ["vertical", "horizontal"]:
                    input_x = upward_pass[stack]
                    x_symmetric = layer_stacks[stack].pop()

                    x = activation(input_x)
                    x = Conv2D(
                        kernel_size=kernel_sizes[stack],
                        kernel_constraint=kernel_constraints[stack],
                    )(x)

                    # Include the vertical-stack layer of the upward pass in the layers
                    # to be added to the horizontal layer.
                    if stack == "horizontal":
                        x_symmetric = keras.layers.Concatenate(axis=-1, dtype=dtype)(
                            [upward_pass["vertical"], x_symmetric]
                        )

                    # Add a skip-connection from the symmetric layer in the downward
                    # pass to the layer `x` in the upward pass.
                    h = activation(x_symmetric)
                    h = Dense(self._num_filters)(h)
                    x = keras.layers.add([h, x], dtype=dtype)

                    x = activation(x)
                    x = keras.layers.Dropout(self._dropout_p, dtype=dtype)(x)
                    x = Conv2D(
                        filters=2 * self._num_filters,
                        kernel_size=kernel_sizes[stack],
                        kernel_constraint=kernel_constraints[stack],
                    )(x)

                    if conditional_input is not None:
                        h_projection = _build_and_apply_h_projection(conditional_input, self._num_filters, dtype=dtype)
                        x = keras.layers.add([x, h_projection], dtype=dtype)

                    x = _apply_sigmoid_gating(x)
                    upward_pass[stack] = keras.layers.add([input_x, x], dtype=dtype)

            # Define deconvolutional layers that expand height/width dimensions on the
            # upward pass (e.g. expanding from 8x8 to 16x16 in Figure 2 of [1]), with
            # the correct kernel dimensions/masking to maintain the autoregressive
            # property.
            if i < self._num_hierarchies - 1:
                for stack in ["vertical", "horizontal"]:
                    h, w = kernel_valid_dims[stack]
                    kernel_height = 2 * h - 2
                    if stack == "vertical":
                        kernel_width = w + 1
                        kernel_constraint = _make_kernel_constraint(
                            (kernel_height, kernel_width),
                            (h - 2, kernel_height),
                            (0, w),
                        )
                    else:
                        kernel_width = 2 * w - 2
                        kernel_constraint = _make_kernel_constraint(
                            (kernel_height, kernel_width),
                            (h - 2, kernel_height),
                            (w - 2, kernel_width),
                        )

                    x = upward_pass[stack]
                    x = Conv2DTranspose(
                        kernel_size=(kernel_height, kernel_width),
                        kernel_constraint=kernel_constraint,
                    )(x)
                    upward_pass[stack] = x

        x_out = keras.layers.ELU(dtype=dtype)(upward_pass["horizontal"])

        # Build final Dense/Reshape layers to output the correct number of
        # parameters per pixel.
        num_channels = tfp_internal.tensorshape_util.as_list(image_shape)[-1]
        num_coeffs = num_channels * (num_channels - 1) // 2  # alpha, beta, gamma in eq.3 of paper
        num_out = num_channels * 2 + num_coeffs + 1  # mu, s + alpha, beta, gamma + 1 (mixture weight)
        num_out_total = num_out * self._num_logistic_mix
        params = Dense(num_out_total)(x_out)
        params = tf.reshape(
            params,
            tfp_internal.prefer_static.concat(  # [-1,H,W,nb mixtures, params per mixture]
                [[-1], image_shape[:-1], [self._num_logistic_mix, num_out]], axis=0
            ),
        )

        # If there is one color channel, split the parameters into a list of three
        # output `Tensor`s: (1) component logits for the Quantized Logistic mixture
        # distributions.distribution, (2) location parameters for each component, and (3) scale
        # parameters for each component. If there is more than one color channel,
        # return a fourth `Tensor` for the coefficients for the linear dependence
        # among color channels (e.g. alpha, beta, gamma).
        # [logits, mu, s, linear dependence]
        splits = 3 if num_channels == 1 else [1, num_channels, num_channels, num_coeffs]
        outputs = tf.split(params, splits, axis=-1)

        # Squeeze singleton dimension from component logits
        outputs[0] = tf.squeeze(outputs[0], axis=-1)

        # Ensure scales are positive and do not collapse to near-zero
        outputs[2] = tf.nn.softplus(outputs[2]) + tf.cast(tf.exp(-7.0), self.dtype)

        inputs = image_input if conditional_input is None else [image_input, conditional_input]
        self._network = keras.Model(inputs=inputs, outputs=outputs)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None, mask: tf.Tensor | None = None) -> tf.Tensor:
        """Call the Pixel CNN network model.

        Parameters
        ----------
        inputs
            4D `Tensor` of image data with dimensions [batch size, height,
            width, channels] or a 2-element `list`. If `list`, the first element is
            the 4D image `Tensor` and the second element is a `Tensor` with
            conditional input data (e.g. VAE encodings or class labels) with the
            same leading batch dimension as the image `Tensor`.
        training
            `bool` or `None`. If `bool`, it controls the dropout layer,
            where `True` implies dropout is active. If `None`, it it defaults to
            `keras.backend.learning_phase()`

        Returns
        -------
        outputs
            a 3- or 4-element `list` of `Tensor`s in the following order: \
            component_logits: 4D `Tensor` of logits for the Categorical distributions.distribution \
            over Quantized Logistic mixture components. Dimensions are \
            `[batch_size, height, width, num_logistic_mix]`.
        locs
            4D `Tensor` of location parameters for the Quantized Logistic \
            mixture components. Dimensions are `[batch_size, height, width, \
            num_logistic_mix, num_channels]`.
        scales
            4D `Tensor` of location parameters for the Quantized Logistic \
            mixture components. Dimensions are `[batch_size, height, width, \
            num_logistic_mix, num_channels]`.
        coeffs
            4D `Tensor` of coefficients for the linear dependence among \
            color channels, included only if the image has more than one channel. \
            Dimensions are `[batch_size, height, width, num_logistic_mix, \
            num_coeffs]`, where `num_coeffs = num_channels * (num_channels - 1) // 2`.
        """
        return self._network(inputs, training=training)


def _make_kernel_constraint(kernel_size, valid_rows, valid_columns):
    """Make the masking function for layer kernels."""
    mask = np.zeros(kernel_size)
    lower, upper = valid_rows
    left, right = valid_columns
    mask[lower:upper, left:right] = 1.0
    mask = mask[:, :, np.newaxis, np.newaxis]
    return lambda x: x * mask


def _build_and_apply_h_projection(h, num_filters, dtype):
    """Project the conditional input."""
    h = keras.layers.Flatten(dtype=dtype)(h)
    h_projection = keras.layers.Dense(2 * num_filters, kernel_initializer="random_normal", dtype=dtype)(h)
    return h_projection[..., tf.newaxis, tf.newaxis, :]


def _apply_sigmoid_gating(x):
    """Apply the sigmoid gating in Figure 2 of [2]."""
    activation_tensor, gate_tensor = tf.split(x, 2, axis=-1)
    sigmoid_gate = tf.sigmoid(gate_tensor)
    return keras.layers.multiply([sigmoid_gate, activation_tensor], dtype=x.dtype)
