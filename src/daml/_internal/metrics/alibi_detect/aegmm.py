from typing import Optional

import alibi_detect
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from daml._internal.metrics.alibi_detect.base import _AlibiDetectMetric


class AlibiAEGMM(_AlibiDetectMetric):
    """
    Gaussian Mixture Model Autoencoder-based outlier detector,
    using alibi-detect aegmm. `<https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/aegmm.html>`_
    """  # noqa E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=alibi_detect.od.OutlierAEGMM,
            flatten_dataset=True,
            dataset_type=np.float32,
        )

    def set_model(
        self,
        encoder_net: tf.keras.Model,
        decoder_net: tf.keras.Model,
        gmm_density_net: tf.keras.Model,
        n_gmm: int,
    ) -> None:
        """
        Sets additional arguments to be used during model creation.

        Note
        ----
        Visit `alibi-detect aegmm <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/aegmm.html#Initialize> for additional information on model parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._model_kwargs, **locals())

    def set_prediction_args(
        self,
        return_instance_score: Optional[bool] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect aegmm <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/aegmm.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    def _get_default_model_kwargs(self) -> dict:
        n_features = tf.math.reduce_prod(self._input_shape)
        latent_dim = 1
        n_gmm = 2  # nb of components in GMM

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=(n_features,)),
                Dense(60, activation=tf.nn.tanh),
                Dense(30, activation=tf.nn.tanh),
                Dense(10, activation=tf.nn.tanh),
                Dense(latent_dim, activation=None),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(10, activation=tf.nn.tanh),
                Dense(30, activation=tf.nn.tanh),
                Dense(60, activation=tf.nn.tanh),
                Dense(n_features, activation=None),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(latent_dim + 2,)),
                Dense(10, activation=tf.nn.tanh),
                Dense(n_gmm, activation=tf.nn.softmax),
            ]
        )

        return {
            "encoder_net": encoder_net,
            "decoder_net": decoder_net,
            "gmm_density_net": gmm_density_net,
            "n_gmm": n_gmm,
        }

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "return_instance_score": True,
            "batch_size": 64,
        }
