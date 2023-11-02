import math
from typing import Optional

import alibi_detect
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    InputLayer,
    Reshape,
)
from tensorflow.nn import relu

from daml._internal.metrics.alibi_detect.base import (
    AlibiDetectOutlierType,
    BaseAlibiDetectOD,
)


class AlibiAE(BaseAlibiDetectOD):
    """
    Autoencoder-based outlier detector, using `alibi-detect ae. <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html>`_
    """  # noqa E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=alibi_detect.od.OutlierAE,
            flatten_dataset=False,
            dataset_type=None,
        )

    def set_model(
        self, encoder_net: tf.keras.Model, decoder_net: tf.keras.Model
    ) -> None:
        """
        Sets additional arguments to be used during model creation.

        Note
        ----
        Visit `alibi-detect ae <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Initialize>`_ for additional information on model parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._model_kwargs, **locals())

    def set_prediction_args(
        self,
        outlier_type: Optional[AlibiDetectOutlierType] = None,
        outlier_perc: Optional[float] = None,
        return_feature_score: Optional[bool] = None,
        return_instance_score: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect ae <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    def _get_default_model_kwargs(self) -> dict:
        encoding_dim = 1024

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=self._input_shape),
                Conv2D(64, 4, strides=2, padding="same", activation=relu),
                Conv2D(128, 4, strides=2, padding="same", activation=relu),
                Conv2D(512, 4, strides=2, padding="same", activation=relu),
                Flatten(),
                Dense(encoding_dim),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(encoding_dim,)),
                Dense(4 * 4 * 128),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding="same", activation=relu),
                Conv2DTranspose(64, 4, strides=2, padding="same", activation=relu),
                Flatten(),
                Dense(math.prod(self._input_shape)),
                Reshape(target_shape=self._input_shape),
            ]
        )

        return {"encoder_net": encoder_net, "decoder_net": decoder_net}

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": AlibiDetectOutlierType.INSTANCE,
            "outlier_perc": 75,
            "return_feature_score": True,
            "return_instance_score": True,
            "batch_size": 64,
        }
