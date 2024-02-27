import math
from typing import Tuple

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    InputLayer,
    Reshape,
)
from tensorflow.math import reduce_prod
from tensorflow.nn import relu, softmax, tanh

from daml._alibi_detect.models.tensorflow.autoencoder import AE, AEGMM, VAE, VAEGMM
from daml._alibi_detect.models.tensorflow.pixelcnn import PixelCNN


def _get_default_encoder_net(input_shape: Tuple[int, int, int], encoding_dim: int):
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv2D(64, 4, strides=2, padding="same", activation=relu),
            Conv2D(128, 4, strides=2, padding="same", activation=relu),
            Conv2D(512, 4, strides=2, padding="same", activation=relu),
            Flatten(),
            Dense(encoding_dim),
        ]
    )


def _get_default_decoder_net(input_shape: Tuple[int, int, int], encoding_dim: int):
    return Sequential(
        [
            InputLayer(input_shape=(encoding_dim,)),
            Dense(4 * 4 * 128),
            Reshape(target_shape=(4, 4, 128)),
            Conv2DTranspose(256, 4, strides=2, padding="same", activation=relu),
            Conv2DTranspose(64, 4, strides=2, padding="same", activation=relu),
            Flatten(),
            Dense(math.prod(input_shape)),
            Reshape(target_shape=input_shape),
        ]
    )


def create_model(
    class_name: str,
    input_shape: Tuple[int, int, int],
    encoding_dim: int = 1024,
    n_gmm: int = 2,
    aegmm_latent_dim: int = 1,
    vaegmm_latent_dim: int = 2,
):
    if class_name == "OD_AE":
        return AE(
            _get_default_encoder_net(input_shape, encoding_dim),
            _get_default_decoder_net(input_shape, encoding_dim),
        )

    if class_name == "OD_VAE":
        return VAE(
            _get_default_encoder_net(input_shape, encoding_dim),
            _get_default_decoder_net(input_shape, encoding_dim),
            encoding_dim,
        )

    if class_name == "OD_AEGMM":
        n_features = reduce_prod(input_shape)
        n_gmm = 2  # nb of components in GMM
        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=(n_features,)),
                Dense(60, activation=tanh),
                Dense(30, activation=tanh),
                Dense(10, activation=tanh),
                Dense(aegmm_latent_dim, activation=None),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(aegmm_latent_dim,)),
                Dense(10, activation=tanh),
                Dense(30, activation=tanh),
                Dense(60, activation=tanh),
                Dense(n_features, activation=None),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(aegmm_latent_dim + 2,)),
                Dense(10, activation=tanh),
                Dense(n_gmm, activation=softmax),
            ]
        )
        return AEGMM(
            encoder_net=encoder_net,
            decoder_net=decoder_net,
            gmm_density_net=gmm_density_net,
            n_gmm=n_gmm,
        )

    if class_name == "OD_VAEGMM":
        n_features = reduce_prod(input_shape)

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=(n_features,)),
                Dense(20, activation=relu),
                Dense(15, activation=relu),
                Dense(7, activation=relu),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(vaegmm_latent_dim,)),
                Dense(7, activation=relu),
                Dense(15, activation=relu),
                Dense(20, activation=relu),
                Dense(n_features, activation=None),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(vaegmm_latent_dim + 2,)),
                Dense(10, activation=relu),
                Dense(n_gmm, activation=softmax),
            ]
        )
        return VAEGMM(
            encoder_net=encoder_net,
            decoder_net=decoder_net,
            gmm_density_net=gmm_density_net,
            n_gmm=n_gmm,
            latent_dim=vaegmm_latent_dim,
        )

    if class_name == "OD_LLR":
        return PixelCNN(
            image_shape=input_shape,
            num_resnet=5,
            num_hierarchies=2,
            num_filters=32,
            num_logistic_mix=1,
            receptive_field_dims=(3, 3),
            dropout_p=0.3,
            l2_weight=0.0,
        )

    raise TypeError("Unknown outlier detection class specified: ", class_name)
