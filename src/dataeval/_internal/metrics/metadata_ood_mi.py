from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif


def get_metadata_ood_mi(
    metadata: Dict[str, Union[List, NDArray]], is_ood: NDArray[np.bool_], discrete_features=None
) -> Dict:
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
    metadata:
        A set of arrays of values, indexed by metadata feature names, with one value per data example per feature.
    is_ood:
        A boolean array, with one value per example, that indicates which examples are OOD.
    discrete_features:
        Either a boolean array or a single boolean value, indicate which features take on discrete values.

    Returns
    -------
    Dict[str, float]
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

    # from dataeval._internal.metrics.utils import infer_categorical

    nats2bits = 1.442695
    discrete_features = False if discrete_features is None else discrete_features
    mdict = metadata

    X = np.array([np.array(v) for v in mdict.values()]).T

    X0, dX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
    Xscl = (X - X0) / dX
    # discrete_features = infer_categorical(Xscl)

    MI = mutual_info_classif(Xscl, is_ood, discrete_features=discrete_features) * nats2bits

    MI_dict = {}
    for i, k in enumerate(mdict):
        MI_dict.update({k: MI[i]})

    return MI_dict
