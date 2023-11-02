"""
This module contains implementations of Image Outlier Detection methods
created by Alibi Detect
"""

from typing import Any, Optional

import alibi_detect
import numpy as np
from alibi_detect.models.tensorflow import PixelCNN

from daml._internal.metrics.alibi_detect.base import (
    AlibiDetectOutlierType,
    BaseAlibiDetectOD,
)


class AlibiLLR(BaseAlibiDetectOD):
    """
    Log likelihood Ratio (LLR) outlier detector,
    using `alibi-detect llr. <https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_llr_mnist.html>`_
    """  # noqa E501

    def __init__(self):
        super().__init__(
            alibi_detect_class=alibi_detect.od.LLR,
            flatten_dataset=False,
            dataset_type=np.float32,
        )

    def set_model(self, model: Any) -> None:
        """
        Sets additional arguments to be used during model creation.

        Note
        ----
        Visit `alibi-detect llr <https://docs.seldon.io/projects/alibi-detect/en/latest/od/methods/llr.html#Initialize>`_ for additional information on model parameters.
        """  # noqa E501
        self._update_kwargs_with_locals(self._model_kwargs, **locals())

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

    def _get_default_model_kwargs(self) -> dict:
        llr_model = PixelCNN(
            image_shape=self._input_shape,
            num_resnet=5,
            num_hierarchies=2,
            num_filters=32,
            num_logistic_mix=1,
            receptive_field_dims=(3, 3),
            dropout_p=0.3,
            l2_weight=0.0,
        )
        return {"model": llr_model}

    @property
    def _default_predict_kwargs(self) -> dict:
        return {
            "outlier_type": AlibiDetectOutlierType.INSTANCE,
            "return_instance_score": True,
            "batch_size": 64,
        }
