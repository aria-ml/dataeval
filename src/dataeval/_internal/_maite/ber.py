import maite.protocols as mp
import numpy as np

import dataeval._internal.metrics.ber as ber


def create_protocol_error_msg(name, protocol):
    return f"{name} expected to be of type maite.protocols.{protocol}"


class BER(ber.BER):
    def __init__(self, data: mp.ArrayLike, labels: mp.ArrayLike, method: ber._METHODS, k: int):
        if not isinstance(data, mp.ArrayLike):
            raise TypeError(create_protocol_error_msg("Data", "ArrayLike"))
        if not isinstance(labels, mp.ArrayLike):
            raise TypeError(create_protocol_error_msg("Labels", "ArrayLike"))

        super().__init__(np.asarray(data), np.asarray(labels), method, k)
