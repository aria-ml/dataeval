import numbers
from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_least_likely_features(
    metadata: dict[str, NDArray], newmetadata: dict[str, Union[list, NDArray]], is_ood: NDArray[np.bool_]
) -> list[tuple[str, float]]:
    """Computes which metadata feature is most out-of-distribution (OOD) relative to a reference metadata set.

    Given a reference metadata dictionary `metadata` (where each key maps to one scalar metadata feature), a second
    metadata dictionary, and a corresponding boolean flag `is_ood` indicating whether each example falls
    out-of-distribution (OOD) relative to the reference, this function finds which metadata feature is the most OOD,
    for each OOD example.

    Parameters
    ----------
    metadata: dict[str, NDArray]
        A reference set of arrays of values, indexed by metadata feature names, with one value per data example per
        feature.
    corrmetadata: dict[str, Union[list, NDArray]]
        A second metedata set, to be tested against the reference metadata. It is ok if the two meta data objects
        hold different numbers of examples.
    is_ood: NDArray[np.bool_]
        A boolean array, with one value per corrmetadata example, that indicates which examples are OOD.

    Returns
    -------
    list[tuple[str, float]]
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
    md_lengths = np.asarray([len(np.atleast_1d(np.asarray(v))) for v in metadata.values()])
    newmd_lengths = np.asarray([len(np.atleast_1d(np.asarray(v))) for v in newmetadata.values()])

    # Raise errors for bad inputs...
    if any(md_lengths < 3):
        raise ValueError("Not enough reference metadata")
        # return [("not enough reference metadata", np.nan)]

    if not all(md_lengths == md_lengths[0]) or not all(newmd_lengths == newmd_lengths[0]):
        raise ValueError("All features must have same length")

    if newmd_lengths[0] != len(is_ood):
        raise ValueError("is_ood flag must have same length as metadata.")

    if np.sum(is_ood) == 0:
        return []
    # ...inputs are good, look for most deviant standardized features.

    # largest standardized absolute deviation from the median observed so far for each example
    deviation = np.zeros_like(is_ood, dtype=np.float32)

    # name of feature that corresponds to `deviation` for each example
    kmax = np.empty(len(is_ood), dtype=object)

    for k, v in metadata.items():
        # exclude cases where random happens to be out on tails, not interesting.
        if k == "random":
            continue

        # Skip non-numerical features
        if not all(isinstance(vi, numbers.Number) for vi in v):  # NB: np.nan *is* a number in this context.
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
