__version__ = "0.0.0"

from importlib.util import find_spec

_IS_TORCH_AVAILABLE = find_spec("torch") is not None
_IS_TENSORFLOW_AVAILABLE = find_spec("tensorflow") is not None and find_spec("tensorflow_probability") is not None

del find_spec
import apipkg

apipkg.initpkg(
    __name__,
    {
        "detectors": {
            "drift": {
                "DriftCVM": "dataeval._internal.detectors.drift:DriftCVM",
                "DriftKS": "dataeval._internal.detectors.drift:DriftKS",
                "DriftOutput": "dataeval._internal.detectors.drift:DriftOutput",
            },
            "linters": {
                "Clusterer": "dataeval._internal.detectors.clusterer:Clusterer",
                "ClustererOutput": "dataeval._internal.detectors.clusterer:Clusterer",
                "Duplicates": "dataeval._internal.detectors.duplicates:Duplicates",
                "DuplicatesOutput": "dataeval._internal.detectors.duplicates:DuplicatesOutput",
                "Outliers": "dataeval._internal.detectors.outliers:Outliers",
                "OutliersOutput": "dataeval._internal.detectors.outliers:OutliersOutput",
            },
        }
    },
)
# from . import detectors, metrics

# __all__ = ["detectors", "metrics"]

# if _IS_TORCH_AVAILABLE:  # pragma: no cover
#     from . import workflows

#     __all__ += ["workflows"]

# if _IS_TENSORFLOW_AVAILABLE or _IS_TORCH_AVAILABLE:  # pragma: no cover
#     from . import utils

#     # __all__ += ["utils"]
