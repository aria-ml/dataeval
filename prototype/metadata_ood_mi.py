from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif

# NATS2BITS is the reciprocal of natural log of 2. If you have an information/entropy-type quantity measured in nats,
#   which is what many library functions return, multiply it by NATS2BITS to get it in bits.
NATS2BITS = 1.442695


def get_metadata_ood_mi(
    metadata: dict[str, list[Any] | NDArray[Any]],
    is_ood: NDArray[np.bool_],
    discrete_features: str | bool | NDArray[np.bool_] = False,
) -> dict[str, float]:
    """Computes mutual information between a set of metadata features and an out-of-distribution flag.

    Given a metadata dictionary `metadata` (where each key maps to one scalar metadata feature per example), and a
    corresponding boolean flag `is_ood` indicating whether each example falls out-of-distribution (OOD) relative to a
    reference dataset, this function finds the strength of association between each metadata feature and `is_ood` by
    computing their mutual information. Metadata features may be either discrete or continuous; set the
    `discrete_features` keyword to a bool array set to True for each feature that is discrete, or pass one bool to apply
    to all features. Returns a dict indicating the strength of association between each individual feature and the OOD
    flag, measured in bits.

    Parameters
    ----------
    metadata : dict[str, list[Any] | NDArray[Any]]
        A set of arrays of values, indexed by metadata feature names, with one value per data example per feature.
    is_ood : NDArray[np.bool_]
        A boolean array, with one value per example, that indicates which examples are OOD.
    discrete_features : str | bool | NDArray[np.bool_]
        Either a boolean array or a single boolean value, indicate which features take on discrete values.

    Returns
    -------
    dict[str, float]
        A dictionary with keys corresponding to metadata feature names, and values indicating the strength of
        association between each named feature and the OOD flag, as mutual information measured in bits.

    Examples
    --------
    Imagine we have 3 data examples, and that the corresponding metadata contains 2 features called time and altitude.

        >>> import numpy
        >>> metadata = {"time": numpy.linspace(0, 10, 100), "altitude": numpy.linspace(0, 16, 100) ** 2}
        >>> is_ood = metadata["altitude"] > 100
        >>> print(get_metadata_ood_mi(metadata, is_ood, discrete_features=False))
    {'time': 0.9407686591507002, 'altitude': 0.9407686591507002}
    """
    mdict = metadata

    X = np.array(list(mdict.values())).T

    X0, dX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
    Xscl = (X - X0) / dX

    mutual_info_values = (
        mutual_info_classif(
            Xscl,
            is_ood,
            discrete_features=discrete_features,  # type: ignore
        )
        * NATS2BITS
    )

    mi_dict = {k: mutual_info_values[i] for i, k in enumerate(mdict)}
    return mi_dict
