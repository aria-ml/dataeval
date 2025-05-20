from __future__ import annotations

__all__ = []

import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif

from dataeval.config import get_seed
from dataeval.data import Metadata
from dataeval.metadata._utils import _compare_keys, _validate_factors_and_data
from dataeval.outputs import MostDeviatedFactorsOutput, OODOutput, OODPredictorOutput
from dataeval.outputs._base import set_metadata


def _calc_median_deviations(reference: NDArray, test: NDArray) -> NDArray:
    """
    Calculates deviations of the test data from the median of the reference data

    Parameters
    ----------
    reference : NDArray
        Reference values of shape (samples, factors)
    test : NDArray
        Incoming values where each sample's factors will be compared to the median of
        the reference set corresponding factors

    Returns
    -------
    NDArray
        Scaled positive and negative deviations of the test data from the reference.

    Note
    ----
    All return values are in the range [0, pos_inf]
    """

    # Take median over samples (rows)
    ref_median = np.median(reference, axis=0)  # (F, )

    # Shift reference and test distributions by reference
    ref_dev = reference - ref_median  # (S, F) - F
    test_dev = test - ref_median  # (S_t, F) - F

    # Separate positive and negative distributions
    # Fills with nans to keep shape in both 1-D and N-D matrices
    pdev = np.where(ref_dev > 0, ref_dev, np.nan)  # (S, F)
    ndev = np.where(ref_dev < 0, ref_dev, np.nan)  # (S, F)

    # Calculate middle of positive and negative distributions per feature
    pscale = np.nanmedian(pdev, axis=0)  # (F, )
    nscale = np.abs(np.nanmedian(ndev, axis=0))  # (F, )

    # Replace 0's for division. Negatives should not happen
    pscale = np.where(pscale > 0, pscale, 1.0)  # (F, )
    nscale = np.where(nscale > 0, nscale, 1.0)  # (F, )

    # Scales positive values by positive scale and negative values by negative
    return np.abs(np.where(test_dev >= 0, test_dev / pscale, test_dev / nscale))  # (S_t, F)


@set_metadata
def find_most_deviated_factors(
    metadata_ref: Metadata,
    metadata_tst: Metadata,
    ood: OODOutput,
) -> MostDeviatedFactorsOutput:
    """
    Determine greatest deviation in metadata features per out of distribution sample in test metadata.

    Parameters
    ----------
    metadata_ref : Metadata
        A reference set of Metadata containing factor names and samples
        with discrete and/or continuous values per factor
    metadata_tst : Metadata
        The set of Metadata that is tested against the reference metadata.
        This set must have the same number of features but does not require the same number of samples.
    ood : OODOutput
        A class output by DataEval's OOD functions that contains which examples are OOD.

    Returns
    -------
    MostDeviatedFactorsOutput
        An output class containing the factor name and deviation of the highest metadata deviations for each
        OOD example in the test metadata.

    Notes
    -----
    1. Both :class:`.Metadata` inputs must have discrete and continuous data in the shape (samples, factors)
       and have equivalent factor names and lengths
    2. The flag at index `i` in :attr:`.OODOutput.is_ood` must correspond
       directly to sample `i` of `metadata_tst` being out-of-distribution from `metadata_ref`

    Examples
    --------

    >>> from dataeval.detectors.ood import OODOutput

    All samples are out-of-distribution

    >>> is_ood = OODOutput(np.array([True, True, True]), np.array([]), np.array([]))
    >>> find_most_deviated_factors(metadata1, metadata2, is_ood)
    MostDeviatedFactorsOutput([('time', 2.0), ('time', 2.592), ('time', 3.51)])

    No samples are out-of-distribution

    >>> is_ood = OODOutput(np.array([False, False, False]), np.array([]), np.array([]))
    >>> find_most_deviated_factors(metadata1, metadata2, is_ood)
    MostDeviatedFactorsOutput([])
    """

    ood_mask: NDArray[np.bool_] = ood.is_ood

    # No metadata correlated with out of distribution data
    if not any(ood_mask):
        return MostDeviatedFactorsOutput([])

    factor_names = metadata_ref.factor_names
    ref_data = metadata_ref.factor_data
    tst_data = metadata_tst.factor_data

    _compare_keys(factor_names, metadata_tst.factor_names)
    _validate_factors_and_data(factor_names, ref_data)
    _validate_factors_and_data(factor_names, tst_data)

    if len(ref_data) < 3:
        warnings.warn(
            f"At least 3 reference metadata samples are needed, got {len(ref_data)}",
            UserWarning,
        )
        return MostDeviatedFactorsOutput([])

    if len(tst_data) != len(ood_mask):
        raise ValueError(
            f"ood and test metadata must have the same length, got {len(ood_mask)} and {len(tst_data)} respectively."
        )

    # Calculates deviations of all samples in m2_data
    # from the median values of the corresponding index in m1_data
    # Guaranteed for inputs to not be empty
    deviations = _calc_median_deviations(ref_data, tst_data)

    # Get most impactful factor deviation of each sample for ood samples only
    deviation = np.max(deviations, axis=1)[ood_mask].astype(np.float16)

    # Get indices of most impactful factors for ood samples only
    max_factors = np.argmax(deviations, axis=1)[ood_mask]

    # Get names of most impactful factors TODO: Find better way than np.dtype(<U4)
    most_ood_factors = np.array(factor_names)[max_factors].tolist()

    # List of tuples matching the factor name with its deviation

    return MostDeviatedFactorsOutput([(factor, dev) for factor, dev in zip(most_ood_factors, deviation)])


_NATS2BITS = 1.442695
"""
_NATS2BITS is the reciprocal of natural log of 2. If you have an information/entropy-type quantity measured in nats,
which is what many library functions return, multiply it by _NATS2BITS to get it in bits.
"""


@set_metadata
def find_ood_predictors(
    metadata: Metadata,
    ood: OODOutput,
) -> OODPredictorOutput:
    """Computes mutual information between a set of metadata features and per sample out-of-distribution flags.

    Given a set of metadata features per sample and a corresponding OODOutput that indicates whether a sample was
    determined to be out of distribution, this function calculates the mutual information between each factor and being
    out of distribution. In other words, it finds which metadata factors most likely correlate to an
    out of distribution sample.

    Note
    ----
    A high mutual information between a factor and ood samples is an indication of correlation, but not causation.
    Additional analysis should be done to determine how to handle factors with a high mutual information.


    Parameters
    ----------
    metadata : Metadata
        A set of arrays of values, indexed by metadata feature names, with one value per data example per feature.
    ood : OODOutput
        A class output by DataEval's OOD functions that contains which examples are OOD.

    Returns
    -------
    OODPredictorOutput
        A dictionary with keys corresponding to metadata feature names, and values indicating the strength of
        association between each named feature and the OOD flag, as mutual information measured in bits.

    Examples
    --------
    >>> from dataeval.outputs import OODOutput

    All samples are out-of-distribution

    >>> is_ood = OODOutput(np.array([True, True, True]), np.array([]), np.array([]))
    >>> find_ood_predictors(metadata1, is_ood)
    OODPredictorOutput({'time': 8.008566032557951e-17, 'altitude': 8.008566032557951e-17})

    No out-of-distribution samples

    >> is_ood = OODOutput(np.array([False, False, False]), np.array([]), np.array([]))
    >> find_ood_predictors(metadata1, is_ood)
    OODPredictorOutput({})
    """

    ood_mask: NDArray[np.bool_] = ood.is_ood

    factors = metadata.factor_names
    data = metadata.factor_data

    # No metadata correlated with out of distribution data, return 0.0 for all factors
    if not any(ood_mask):
        return OODPredictorOutput(dict.fromkeys(factors, 0.0))

    if len(data) != len(ood_mask):
        raise ValueError(
            f"ood and metadata must have the same length, got {len(ood_mask)} and {len(data)} respectively."
        )

    # Calculate mean, std of each factor over all samples
    scaled_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)  # (S, F)

    discrete_features = [info.factor_type != "continuous" for info in metadata.factor_info.values()]

    mutual_info_values = (
        mutual_info_classif(
            X=scaled_data,
            y=ood_mask,
            discrete_features=discrete_features,  # type: ignore - sklearn function not typed
            random_state=get_seed(),
        )
        * _NATS2BITS
    )

    return OODPredictorOutput({k: mutual_info_values[i] for i, k in enumerate(factors)})
