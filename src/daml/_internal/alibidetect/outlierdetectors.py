import warnings
from abc import ABC
from typing import Any, Dict, Iterable, Sequence, Union

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

from daml._internal.MetricClasses import OutlierDetector


class AlibiDetectODMetric(OutlierDetector, ABC):
    """Abstract class for all outlier detection metrics in alibi-detect"""

    def __init__(self):
        super().__init__()
        self.detector: Any = None

    # Train the alibi-detect metric on dataset
    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int,
        verbose: bool,
    ) -> None:
        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )
        self.detector.fit(dataset, epochs=epochs, verbose=verbose)
        # if save_path: save_detector(self.detector, save_path)
        self.is_trained: bool = True


class AlibiAE(AlibiDetectODMetric):
    """Autoencoder-based outlier detector, from alibi-detect"""

    def __init__(self):
        super().__init__()
        # if load_path and detector_type and dataset_name and detector_name:
        #     self.detector = fetch_detector(detector, save_path)
        self.detector: Any = self.initialize_detector()

    def initialize_detector(self) -> tf.keras.Sequential:
        """Initialize the architecture and model weights of the autoencoder"""
        tf.keras.backend.clear_session()
        encoding_dim = 1024

        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(
                    64,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2D(
                    128,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2D(
                    512,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
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
                Conv2DTranspose(
                    256,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2DTranspose(
                    64,
                    4,
                    strides=2,
                    padding="same",
                    activation=relu,
                ),
                Conv2DTranspose(
                    3,
                    4,
                    strides=2,
                    padding="same",
                    activation="sigmoid",
                ),
            ]
        )

        # initialize outlier detector using autoencoder network
        detector = alibi_detect.od.OutlierAE(
            threshold=0.015,  # threshold for outlier score
            encoder_net=encoder_net,  # can also pass AE model instead
            decoder_net=decoder_net,  # of separate encoder and decoder
        )
        return detector

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """Train the outlier detector on dataset"""

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(
        self, dataset: Iterable[float]
    ) -> Dict[str, Dict[str, Union[str, Sequence[float]]]]:
        """Evaluate the outlier detector metric on dataset"""

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            warnings.warn(
                "Warning: Evaluating a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )
        predictions = self.detector.predict(
            dataset,
            outlier_type="instance",
            outlier_perc=75,
            return_feature_score=True,
            return_instance_score=True,
        )
        return predictions
