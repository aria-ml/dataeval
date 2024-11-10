"""
TensorFlow models are used in :term:`out of distribution<Out-of-distribution (OOD)>` detectors in the
:mod:`dataeval.detectors.ood` module.

DataEval provides basic default models through the utility :func:`dataeval.utils.tensorflow.create_model`.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE

__all__ = []


if _IS_TENSORFLOW_AVAILABLE:
    import dataeval.utils.tensorflow.loss as loss
    from dataeval.utils.tensorflow._internal.utils import create_model

    __all__ = ["create_model", "loss"]

del _IS_TENSORFLOW_AVAILABLE
