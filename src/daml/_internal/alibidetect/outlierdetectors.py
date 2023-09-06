from abc import ABC
from typing import Any, Iterable, Optional, Type

import alibi_detect
import numpy as np
import tensorflow as tf
from alibi_detect.models.tensorflow import PixelCNN
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
from daml._internal.MetricOutputs import AlibiOutlierDetectorOutput


class AlibiDetectODMetric(OutlierDetector, ABC):
    """Abstract class for all outlier detection metrics in alibi-detect"""

    def __init__(self):
        super().__init__()
        self.detector: Any = None
        self._kwargs = dict()
        # if load_path and detector_type and dataset_name and detector_name:
        #     self.detector = fetch_detector(detector, save_path)
        self.detector: Any = self.initialize_detector()

    # Train the alibi-detect metric on dataset
    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int,
        verbose: bool,
    ) -> None:
        """"""

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        self.detector.fit(dataset, epochs=epochs, verbose=verbose)
        # if save_path: save_detector(self.detector, save_path)
        self.is_trained: bool = True

    def format_results(self, preds):
        output = AlibiOutlierDetectorOutput(
            is_outlier=preds["data"]["is_outlier"],
            feature_score=preds["data"]["feature_score"],
            instance_score=preds["data"]["instance_score"],
        )
        return output


class AlibiAE(AlibiDetectODMetric):
    """Autoencoder-based outlier detector, from alibi-detect"""

    def __init__(self):
        super().__init__()
        self._DATASET_TYPE: Optional[Type] = None

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

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": 0.015,  # threshold for outlier score
                "encoder_net": encoder_net,  # can also pass AE model instead
                "decoder_net": decoder_net,  # of separate encoder and decoder
            }
        )
        # initialize outlier detector using autoencoder network
        return alibi_detect.od.OutlierAE(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """Train the outlier detector on dataset"""

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(self, dataset: Iterable[float]) -> AlibiOutlierDetectorOutput:
        """Evaluate the outlier detector metric on dataset"""

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )
        predictions = self.detector.predict(
            dataset,
            outlier_type="instance",
            outlier_perc=75,
            return_feature_score=True,
            return_instance_score=True,
        )
        return self.format_results(predictions)


class AlibiVAE(AlibiDetectODMetric):
    """Autoencoder-based outlier detector, from alibi-detect"""

    def __init__(self):
        super().__init__()
        self._DATASET_TYPE: Optional[Type] = None

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

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": 0.015,  # threshold for outlier score
                "encoder_net": encoder_net,  # can also pass AE model instead
                "decoder_net": decoder_net,  # of separate encoder and decoder
                "latent_dim": 1024,
                "samples": 10,
            }
        )
        # initialize outlier detector using autoencoder network
        return alibi_detect.od.OutlierVAE(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """Train the outlier detector on dataset"""

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(self, dataset: Iterable[float]) -> AlibiOutlierDetectorOutput:
        """Evaluate the outlier detector metric on dataset"""

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )
        predictions = self.detector.predict(
            dataset,
            outlier_type="instance",
            outlier_perc=75,
            return_feature_score=True,
            return_instance_score=True,
        )
        return self.format_results(predictions)


class AlibiAEGMM(AlibiDetectODMetric):
    """
    Gaussian Mixture Model Autoencoder-based outlier detector,
    from alibi-detect.
    Based on https://docs.seldon.io/projects/alibi-detect/
             en/latest/examples/od_aegmm_kddcup.html
    """

    def __init__(self):
        super().__init__()
        self._FLATTEN_DATASET: bool = True
        self._DATASET_TYPE: Type = np.float32

    def initialize_detector(self) -> tf.keras.Sequential:
        """Initialize the architecture and model weights of the autoencoder"""
        tf.keras.backend.clear_session()
        n_features = 32 * 32 * 3
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

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": None,  # threshold for outlier score
                "encoder_net": encoder_net,  # can also pass AE model instead
                "decoder_net": decoder_net,  # of separate encoder and decoder
                "gmm_density_net": gmm_density_net,
                "n_gmm": n_gmm,
            }
        )
        # initialize outlier detector using autoencoder network
        return alibi_detect.od.OutlierAEGMM(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """Train the outlier detector on dataset"""

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET,
            dataset_type=self._DATASET_TYPE,
        )

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(self, dataset: Iterable[float]) -> AlibiOutlierDetectorOutput:
        """Evaluate the outlier detector metric on dataset"""

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET,
            dataset_type=self._DATASET_TYPE,
        )

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )
        predictions = self.detector.predict(
            dataset,
            return_instance_score=True,
        )
        return self.format_results(predictions)


class AlibiVAEGMM(AlibiDetectODMetric):
    """
    Variational Gaussian Mixture Model Autoencoder-based outlier detector,
    from alibi-detect.
    Based on https://docs.seldon.io/projects/alibi-detect
             /en/latest/od/methods/vaegmm.html
    """

    def __init__(self):
        super().__init__()
        self._FLATTEN_DATASET: bool = True
        self._DATASET_TYPE = np.float32

    def initialize_detector(self) -> tf.keras.Sequential:
        """Initialize the architecture and model weights of the autoencoder"""
        tf.keras.backend.clear_session()
        n_features = 32 * 32 * 3
        latent_dim = 2
        n_gmm = 2  # nb of components in GMM

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

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": None,  # threshold for outlier score
                "encoder_net": encoder_net,  # can also pass AE model instead
                "decoder_net": decoder_net,  # of separate encoder and decoder
                "gmm_density_net": gmm_density_net,
                "n_gmm": n_gmm,
                "latent_dim": latent_dim,
                "samples": 10,
            }
        )
        # initialize outlier detector using autoencoder network
        return alibi_detect.od.OutlierVAEGMM(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """Train the outlier detector on dataset"""

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET,
            dataset_type=self._DATASET_TYPE,
        )

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(dataset, threshold_perc=95)

    def evaluate(self, dataset: Iterable[float]) -> AlibiOutlierDetectorOutput:
        """Evaluate the outlier detector metric on dataset"""

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET,
            dataset_type=self._DATASET_TYPE,
        )

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )
        predictions = self.detector.predict(
            dataset,
            return_instance_score=True,
        )
        return self.format_results(predictions)


class AlibiLLR(AlibiDetectODMetric):
    """
    Log likelihood Ratio (LLR) outlier detector, from alibi-detect
    Based on https://docs.seldon.io/projects/alibi-detect
             /en/latest/examples/od_llr_mnist.html
    """

    def __init__(self):
        super().__init__()
        self._FLATTEN_DATASET = False
        self._DATASET_TYPE: type = np.float32

    def initialize_detector(self) -> tf.keras.Sequential:
        """Initialize the architecture and model weights of the autoencoder"""
        tf.keras.backend.clear_session()
        input_shape = (32, 32, 3)

        # LLR internally uses a Pixel CNN architecture,
        # which we initialize here
        llr_model = PixelCNN(
            image_shape=input_shape,
            num_resnet=5,
            num_hierarchies=2,
            num_filters=32,
            num_logistic_mix=1,
            receptive_field_dims=(3, 3),
            dropout_p=0.3,
            l2_weight=0.0,
        )

        # common inputs across encoders...
        self._kwargs.update(
            {
                "threshold": None,  # threshold for outlier score
                "model": llr_model,
            }
        )
        # initialize outlier detector using autoencoder network
        return alibi_detect.od.LLR(**self._kwargs)

    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """Train the outlier detector on dataset"""

        # Cast and flatten dataset
        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET,
            dataset_type=self._DATASET_TYPE,
        )

        super().fit_dataset(dataset, epochs, verbose)
        self.detector.infer_threshold(
            dataset,
            threshold_perc=95,
            batch_size=32,
        )

    def evaluate(self, dataset: Iterable[float]) -> AlibiOutlierDetectorOutput:
        """Evaluate the outlier detector metric on dataset"""
        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        if not self.is_trained:
            raise TypeError(
                "Error: tried to evaluate a metric that is not trained. \
                    Try calling metric.fit_dataset(data)"
            )

        dataset = self.format_dataset(
            dataset,
            flatten_dataset=self._FLATTEN_DATASET,
            dataset_type=self._DATASET_TYPE,
        )

        predictions = self.detector.predict(
            dataset,
            outlier_type="instance",
            return_instance_score=True,
        )
        return self.format_results(predictions)
