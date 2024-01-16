from typing import Optional

from daml import _alibi_detect
from daml._internal.metrics.alibi_detect.base import (
    AlibiDetectOutlierType,
    _AlibiDetectMetric,
)


class AlibiAE(_AlibiDetectMetric):
    """
    Autoencoder-based outlier detector, using `alibi-detect ae. <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/ae.html>`_

    The model used by this class is :py:class:`daml.models.AE`
    """  # noqa E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.OutlierAE,
            model_param_name="ae",
            flatten_dataset=False,
            dataset_type=None,
        )

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

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": AlibiDetectOutlierType.INSTANCE,
            "outlier_perc": 75,
            "return_feature_score": True,
            "return_instance_score": True,
            "batch_size": 64,
        }
