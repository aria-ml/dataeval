"""
Detectors can determine if a dataset or individual images in a dataset are indicative of a specific issue.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE

from . import drift, linters

__all__ = ["drift", "linters"]

if _IS_TENSORFLOW_AVAILABLE:  # pragma: no cover
    # from . import ood
    # __all__ += ["ood"]
    import apipkg

    apipkg.initpkg(
        __name__,
        {
            "ood": {
                "OOD_AE": "dataeval._internal.detectors.ood.ae:OOD_AE",
                "OOD_AEGMM": "dataeval._internal.detectors.ood.aegmm:OOD_AEGMM",
                "OOD_LLR": "dataeval._internal.detectors.ood.llr:OOD_LLR",
                "OOD_VAE": "dataeval._internal.detectors.ood.vae:OOD_VAE",
                "OOD_VAEGMM": "dataeval._internal.detectors.ood.vae:OOD_VAEGMM",
            }
        },
    )
