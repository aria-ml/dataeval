from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray


def get_least_likely_features(
    metadata: Dict[str, Union[List, NDArray]], corrmetadata: Dict[str, Union[List, NDArray]], is_ood: NDArray[np.bool_]
) -> NDArray[str]:
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

    >>> from metadata_tools import get_least_likely_features
    >>> import numpy
    >>> rng = numpy.random.default_rng(123)
    >>> metadata = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    >>> corrmetadata = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, 211101]}
    >>> is_ood = rng.choice(a=[False, True], size=len(metadata["time"]))
    >>> get_least_likely_features(metadata, corrmetadata, is_ood)
    array(['time', 'time'], dtype='<U4')
    """
    norm_dict = {}
    for k, v in metadata.items():
        loc = np.median(v)
        dev = v - loc
        posdev, negdev = dev[dev > 0], dev[dev < 0]
        pos_scale = np.median(posdev)
        neg_scale = np.abs(np.median(negdev))

        norm_dict.update({k: {"loc": loc, "pos_scale": pos_scale, "neg_scale": neg_scale}})

    mkeys = list(norm_dict)

    maxpdev = np.array([-1e30 for _ in is_ood])
    maxndev = -1.0 * maxpdev

    deviation = np.zeros(is_ood.shape)
    ikmax = np.zeros(is_ood.shape, dtype=np.int32)
    for ik, k in enumerate(mkeys):
        if k == "random":  # exclude cases where random happens to be out on tails, not interesting.
            continue
        ndk = norm_dict[k]
        x, x0, dxp, dxn = corrmetadata[k], ndk["loc"], ndk["pos_scale"], ndk["neg_scale"]
        dxp = dxp if dxp > 0 else 1.0
        dxn = dxn if dxn > 0 else 1.0

        xdev = x - x0
        pos, neg = xdev >= 0, xdev < 0

        X = np.zeros_like(x)
        X[pos], X[neg] = xdev[pos] / dxp, xdev[neg] / dxn

        pbig, nbig, abig = maxpdev < X, maxndev > X, np.abs(X) > deviation

        update_mpdev, update_mndev = np.logical_and(pbig, abig), np.logical_and(nbig, abig)
        maxpdev[update_mpdev], maxndev[update_mndev] = X[update_mpdev], X[update_mndev]

        update_k = np.logical_or(update_mpdev, update_mndev)
        ikmax[update_k] = ik
        deviation[update_k] = np.abs(X[update_k])

    unlikely_features = np.array([mkeys[ik] for ik in ikmax])[is_ood]
    return unlikely_features
