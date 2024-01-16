"""
This module contains implementations of Image Outlier Detection methods
created by Alibi Detect
"""

from typing import Optional

import numpy as np

from daml import _alibi_detect
from daml._internal.metrics.alibi_detect.base import (
    AlibiDetectOutlierType,
    _AlibiDetectMetric,
)


class AlibiLLR(_AlibiDetectMetric):
    """
    Log likelihood Ratio (LLR) outlier detector,
    using `alibi-detect llr. <https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_llr_mnist.html>`_


    The model used by this class is :py:class:`daml.models.LLR`
    """  # noqa E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=_alibi_detect.od.LLR,
            model_param_name="model",
            flatten_dataset=False,
            dataset_type=np.float32,
        )

    def set_prediction_args(
        self,
        outlier_type: Optional[AlibiDetectOutlierType] = None,
        return_instance_score: Optional[bool] = None,
    ) -> None:
        """
        Sets additional arguments to be used during prediction.

        Note
        ----
        Visit `alibi-detect llr <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/llr.html#Detect>`_ for additional information on prediction parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._predict_kwargs, **locals())

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": AlibiDetectOutlierType.INSTANCE,
            "return_instance_score": True,
            "batch_size": 64,
        }
