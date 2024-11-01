from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_least_likely_features(
    metadata: dict[str, NDArray], newmetadata: dict[str, Union[list, NDArray]], is_ood: NDArray[np.bool_]
) -> list[tuple[str, float]]:  # NDArray[np.str_]:
    """Computes which metadata feature is most out-of-distribution (OOD) relative to a reference metadata set.

        Given a reference metadata dictionary `metadata` (where each key maps to one scalar metadata feature), a second
        metadata dictionary, and a corresponding boolean flag `is_ood` indicating whether each example falls
        out-of-distribution (OOD) relative to the reference, this function finds which metadata feature is the most OOD,
        for each OOD example.

        Parameters
        ----------
        metadata:
            A reference set of arrays of values, indexed by metadata feature names, with one value per data example per
            feature.
        corrmetadata:
            A second metedata set, to be tested against the reference metadata. It is ok if the two meta data objects
            hold different numbers of examples.
        is_ood:
            A boolean array, with one value per corrmetadata example, that indicates which examples are OOD.

        Returns
        -------
        NDArray[str]
            An array of names of the features of each OOD corrmetadata example that were the most OOD.

        Examples
        --------
        Imagine we have 3 data examples, and that the corresponding metadata contains 2 features called time and
        altitude, as shown below.

    from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features
    >>> import numpy
    >>> metadata = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    >>> newmetadata = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -211101]}
    >>> is_ood = numpy.array([True, True, True])
    >>> get_least_likely_features(metadata, newmetadata, is_ood)
    array(['time', 'time', 'altitude'], dtype=object)
    """
    # largest standardized absolute deviation from the median observed so far for each example
    deviation = np.zeros_like(is_ood, dtype=np.float32)

    # name of feature that corresponds to `deviation` (see above) for each example
    kmax = np.empty(len(is_ood), dtype=object)

    for k, v in metadata.items():
        if k == "random":  # exclude cases where random happens to be out on tails, not interesting.
            continue

        # Get standardization parameters from metadata
        loc = np.median(v)
        dev = v - loc
        posdev, negdev = dev[dev > 0], dev[dev < 0]
        pos_scale = np.median(posdev) if posdev.any() else 1.0
        neg_scale = np.abs(np.median(negdev)) if negdev.any() else 1.0

        x, x0, dxp, dxn = np.atleast_1d(newmetadata[k]), loc, pos_scale, neg_scale  # just abbreviations
        dxp = dxp if dxp > 0 else 1.0  # avoids dividing by zero below
        dxn = dxn if dxn > 0 else 1.0

        # xdev must be floating-point to avoid getting zero in an integer division.
        xdev = (x - x0).astype(np.float64)
        pos = xdev >= 0

        X = np.zeros_like(xdev)
        X[pos], X[~pos] = xdev[pos] / dxp, xdev[~pos] / dxn  # keeping track of possible asymmetry of x, but...
        # ...below here, only need to think about absolute deviation.
        abig = np.abs(X) > deviation
        kmax[abig] = k
        deviation[abig] = np.abs(X[abig])

    unlikely_features = list(zip(kmax[is_ood], deviation[is_ood]))
    return unlikely_features
