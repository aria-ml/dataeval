"""
Tensorflow models are used in out-of-distribution detectors in the :mod:`dataeval.detectors.ood` module.

DataEval provides both basic default models through the utility :func:`dataeval.utils.tensorflow.models.create_model`
as well as constructors which allow for customization of the encoder, decoder and any other applicable
layers used by the model.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE

from . import loss, models, recon

__all__ = []


if _IS_TENSORFLOW_AVAILABLE:
    __all__ = ["loss", "models", "recon"]
