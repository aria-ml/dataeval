"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from typing import Callable, cast

import keras
import tensorflow as tf
from keras.layers import (
    Dense,
    Flatten,
    Layer,
)


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
        x = cast(tf.Tensor, Flatten()(x))
        y = cast(tf.Tensor, Flatten()(y))
    rec_cos = tf.reshape(keras.losses.cosine_similarity(y, x, -1), (-1, 1))
    rec_euc = tf.reshape(relative_euclidean_distance(y, x, -1), (-1, 1))
    # rec_euc could become very large so should be clipped
    rec_euc = tf.clip_by_value(rec_euc, 0, max_eucl)
    return cast(tf.Tensor, tf.concat([rec_cos, rec_euc], -1))


class Sampling(Layer):
    """Reparametrization trick - Uses (z_mean, z_log_var) to sample the latent vector z."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Sample z.

        Parameters
        ----------
        inputs
            Tuple with mean and log variance.

        Returns
        -------
        Sampled vector z.
        """
        z_mean, z_log_var = inputs
        batch, dim = tuple(tf.shape(z_mean).numpy().ravel()[:2])  # type: ignore
        epsilon = cast(tf.Tensor, keras.backend.random_normal(shape=(batch, dim)))
        return z_mean + tf.exp(tf.math.multiply(0.5, z_log_var)) * epsilon


class EncoderAE(Layer):
    def __init__(self, encoder_net: keras.Model) -> None:
        """
        Encoder of AE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a keras.Sequential class.
        name
            Name of encoder.
        """
        super().__init__(name="encoder_ae")
        self.encoder_net = encoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return cast(tf.Tensor, self.encoder_net(x))


class EncoderVAE(Layer):
    def __init__(self, encoder_net: keras.Model, latent_dim: int) -> None:
        """
        Encoder of VAE.

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a keras.Sequential class.
        latent_dim
            Dimensionality of the latent space.
        name
            Name of encoder.
        """
        super().__init__(name="encoder_vae")
        self.encoder_net = encoder_net
        self.fc_mean = Dense(latent_dim, activation=None)
        self.fc_log_var = Dense(latent_dim, activation=None)
        self.sampling = Sampling()

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = cast(tf.Tensor, self.encoder_net(x))
        if len(x.shape) > 2:
            x = cast(tf.Tensor, Flatten()(x))
        z_mean = cast(tf.Tensor, self.fc_mean(x))
        z_log_var = cast(tf.Tensor, self.fc_log_var(x))
        z = cast(tf.Tensor, self.sampling((z_mean, z_log_var)))
        return z_mean, z_log_var, z


class Decoder(Layer):
    def __init__(self, decoder_net: keras.Model) -> None:
        """
        Decoder of AE and VAE.

        Parameters
        ----------
        decoder_net
            Layers for the decoder wrapped in a keras.Sequential class.
        name
            Name of decoder.
        """
        super().__init__(name="decoder")
        self.decoder_net = decoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return cast(tf.Tensor, self.decoder_net(x))


class AE(keras.Model):
    """
    Combine encoder and decoder in AE.

    Parameters
    ----------
    encoder_net : keras.Model
        Layers for the encoder wrapped in a keras.Sequential class.
    decoder_net : keras.Model
        Layers for the decoder wrapped in a keras.Sequential class.
    """

    def __init__(self, encoder_net: keras.Model, decoder_net: keras.Model) -> None:
        super().__init__(name="ae")
        self.encoder = EncoderAE(encoder_net)
        self.decoder = Decoder(decoder_net)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z = cast(tf.Tensor, self.encoder(x))
        x_recon = cast(tf.Tensor, self.decoder(z))
        return x_recon


class VAE(keras.Model):
    """
    Combine encoder and decoder in VAE.

    Parameters
    ----------
    encoder_net : keras.Model
        Layers for the encoder wrapped in a keras.Sequential class.
    decoder_net : keras.Model
        Layers for the decoder wrapped in a keras.Sequential class.
    latent_dim : int
        Dimensionality of the latent space.
    beta : float, default 1.0
        Beta parameter for KL-divergence loss term.
    """

    def __init__(self, encoder_net: keras.Model, decoder_net: keras.Model, latent_dim: int, beta: float = 1.0) -> None:
        super().__init__(name="vae_model")
        self.encoder = EncoderVAE(encoder_net, latent_dim)
        self.decoder = Decoder(decoder_net)
        self.beta = beta
        self.latent_dim = latent_dim

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var, z = cast(tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.encoder(x))
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
    encoder_net : keras.Model
        Layers for the encoder wrapped in a keras.Sequential class.
    decoder_net : keras.Model
        Layers for the decoder wrapped in a keras.Sequential class.
    gmm_density_net : keras.Model
        Layers for the GMM network wrapped in a keras.Sequential class.
    n_gmm : int
        Number of components in GMM.
    recon_features : Callable, default eucl_cosim_features
        Function to extract features from the reconstructed instance by the decoder.
    """

    def __init__(
        self,
        encoder_net: keras.Model,
        decoder_net: keras.Model,
        gmm_density_net: keras.Model,
        n_gmm: int,
        recon_features: Callable = eucl_cosim_features,
    ) -> None:
        super().__init__("aegmm")
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.gmm_density = gmm_density_net
        self.n_gmm = n_gmm
        self.recon_features = recon_features

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        enc = self.encoder(x)
        x_recon = cast(tf.Tensor, self.decoder(enc))
        recon_features = self.recon_features(x, x_recon)
        z = cast(tf.Tensor, tf.concat([enc, recon_features], -1))
        gamma = cast(tf.Tensor, self.gmm_density(z))
        return x_recon, z, gamma


class VAEGMM(keras.Model):
    """
    Variational Autoencoding Gaussian Mixture Model.

    Parameters
    ----------
    encoder_net : keras.Model
        Layers for the encoder wrapped in a keras.Sequential class.
    decoder_net : keras.Model
        Layers for the decoder wrapped in a keras.Sequential class.
    gmm_density_net : keras.Model
        Layers for the GMM network wrapped in a keras.Sequential class.
    n_gmm : int
        Number of components in GMM.
    latent_dim : int
        Dimensionality of the latent space.
    recon_features : Callable, default eucl_cosim_features
        Function to extract features from the reconstructed instance by the decoder.
    beta : float, default 1.0
        Beta parameter for KL-divergence loss term.
    """

    def __init__(
        self,
        encoder_net: keras.Model,
        decoder_net: keras.Model,
        gmm_density_net: keras.Model,
        n_gmm: int,
        latent_dim: int,
        recon_features: Callable = eucl_cosim_features,
        beta: float = 1.0,
    ) -> None:
        super().__init__(name="vaegmm")
        self.encoder = EncoderVAE(encoder_net, latent_dim)
        self.decoder = decoder_net
        self.gmm_density = gmm_density_net
        self.n_gmm = n_gmm
        self.latent_dim = latent_dim
        self.recon_features = recon_features
        self.beta = beta

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        enc_mean, enc_log_var, enc = cast(tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.encoder(x))
        x_recon = cast(tf.Tensor, self.decoder(enc))
        recon_features = self.recon_features(x, x_recon)
        z = cast(tf.Tensor, tf.concat([enc, recon_features], -1))
        gamma = cast(tf.Tensor, self.gmm_density(z))
        # add KL divergence loss term
        kl_loss = -0.5 * tf.reduce_mean(enc_log_var - tf.square(enc_mean) - tf.exp(enc_log_var) + 1)
        self.add_loss(self.beta * kl_loss)
        return x_recon, z, gamma
