"""
This module contains implementations of Image Outlier Detection methods
created by Alibi Detect
"""

from abc import ABC
from typing import Any, Dict, Iterable, Optional, Tuple

from daml._internal.metrics.base import OutlierDetector
from daml._internal.metrics.outputs import OutlierDetectorOutput


class BaseAlibiDetectOD(OutlierDetector, ABC):
    """
    Base class for all outlier detection metrics in alibi-detect

    Attributes
    ----------
    detector: Any, default None
        A model used for outlier detection after being trained on clean data

    .. todo:: Add model loading & saving
    """

    def __init__(self):
        """Constructor method"""

        super().__init__()
        self._kwargs = dict()
        self.detector: Any = None
        self._reference_input_shape: Optional[Tuple[int, int, int]] = None

    # Train the alibi-detect metric on dataset
    def fit_dataset(
        self,
        dataset: Iterable[float],
        epochs: int = 3,
        verbose: bool = False,
    ) -> None:
        """
        Trains a model on a dataset containing that can be used
        for the detection of outliers in :method:`evaluate`

        Parameters
        ----------
        dataset : Iterable[float]
            An array of images for the model to train on
        epochs : int, default 3
            Number of epochs to train the detector for.
        verbose : bool, default False
            Flag to output logs from Alibi-Detect verbose mode.bi-Detect verbose mode.

        Raises
        ------
        TypeError
            If the detector has not been initialized or loaded from path

        .. note::
            The supplied dataset should contain no outliers for maximum benefit
        """

        if self.detector is None:
            raise TypeError(
                "Tried to evaluate without initializing a detector. \
                    Try calling metric.initialize_detector()"
            )

        self.detector.fit(dataset, epochs=epochs, verbose=verbose)
        # if save_path: save_detector(self.detector, save_path)
        self.is_trained: bool = True

    def _format_results(
        self, preds: Dict[str, Dict[str, Any]]
    ) -> OutlierDetectorOutput:
        """
        Changes outlier detection dictionary outputs into an \
        :class:`OutlierDetectorOutput` dataclass

        Parameters
        ----------
        preds : Dict[str, Dict[str, Any]]
            Dictionary of output data from :class:`BaseAlibiDetectOD` subclasses

        Returns
        -------
        :class:`OutlierDetectorOutput`
            A dataclass containing outlier mask, feature scores
            and instance scores if applicable
        """

        output = OutlierDetectorOutput(
            is_outlier=preds["data"]["is_outlier"],
            feature_score=preds["data"]["feature_score"],
            instance_score=preds["data"]["instance_score"],
        )
        return output
