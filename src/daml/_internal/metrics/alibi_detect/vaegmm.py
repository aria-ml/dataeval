from typing import Any, Optional

import alibi_detect
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from daml._internal.metrics.alibi_detect.base import BaseAlibiDetectOD


class AlibiVAEGMM(BaseAlibiDetectOD):
    """
    Variational Gaussian Mixture Model Autoencoder-based outlier detector, from alibi-detect

    Based on https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/vaegmm.html
    """  # noqa E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=alibi_detect.od.OutlierVAEGMM,
            flatten_dataset=True,
            dataset_type=np.float32,
        )

    def set_model(
        self,
        encoder_net: Any,
        decoder_net: Any,
        gmm_density_net: Any,
        n_gmm: int,
        latent_dim: int,
        samples: int,
    ) -> None:
        self._update_kwargs_with_locals(self._model_kwargs, **locals())

    def set_prediction_args(
        self,
        return_instance_score: Optional[bool] = None,
    ) -> None:
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    def _get_default_model_kwargs(self) -> dict:
        n_features = tf.math.reduce_prod(self._input_shape)
        latent_dim = 2
        n_gmm = 2  # nb of components in GMM
        samples = 10

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=(n_features,)),
                Dense(20, activation=tf.nn.relu),
                Dense(15, activation=tf.nn.relu),
                Dense(7, activation=tf.nn.relu),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(7, activation=tf.nn.relu),
                Dense(15, activation=tf.nn.relu),
                Dense(20, activation=tf.nn.relu),
                Dense(n_features, activation=None),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(latent_dim + 2,)),
                Dense(10, activation=tf.nn.relu),
                Dense(n_gmm, activation=tf.nn.softmax),
            ]
        )

        return {
            "encoder_net": encoder_net,
            "decoder_net": decoder_net,
            "gmm_density_net": gmm_density_net,
            "n_gmm": n_gmm,
            "latent_dim": latent_dim,
            "samples": samples,
        }

    @property
    def _default_predict_kwargs(self) -> dict:
        return {"return_instance_score": True}
