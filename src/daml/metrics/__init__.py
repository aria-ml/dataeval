from importlib.util import find_spec

from daml._internal.metrics.ber import BER
from daml._internal.metrics.divergence import Divergence
from daml._internal.metrics.uap import UAP

__all__ = ["BER", "Divergence", "UAP"]

if (
    find_spec("tensorflow") is not None
    and find_spec("tensorflow_probability") is not None
):  # pragma: no cover
    from . import outlier_detection

    __all__ += ["outlier_detection"]

if find_spec("torch") is not None:  # pragma: no cover
    from daml._internal.metrics.sufficiency import Sufficiency

    __all__ += ["Sufficiency"]

del find_spec
