from __future__ import annotations

__all__ = []

import warnings

import numpy as np
from numpy.typing import NDArray

from dataeval.detectors.ood import OODOutput
from dataeval.metadata._utils import _compare_keys, _validate_factors_and_data
from dataeval.utils.data import Metadata


def _combine_metadata(metadata_1: Metadata, metadata_2: Metadata) -> tuple[list[str], list[NDArray], list[NDArray]]:
    """
    Combines the factor names and data arrays of metadata_1 and metadata_2 when the names
    match exactly and data has the same number of columns (factors).

    Parameters
    ----------
    metadata_1 : Metadata
        The set of factor names used as reference to determine the correct factor names and length of data
    metadata_2 : Metadata
        The compared set of factor names and data that must match metadata_1

    Returns
    -------
    list[str]
        The combined discrete and continuous factor names in that order.
    list[NDArray]
        Combined discrete and continuous data of metadata_1
    list[NDArray]
        Combined discrete and continuous data of metadata_2

    Raises
    ------
    ValueError
        If keys do not match in metadata_1 and metadata_2
    ValueError
        If the length of keys do not match the length of the data
    """
    factor_names: list[str] = []
    m1_data: list[NDArray] = []
    m2_data: list[NDArray] = []

    # Both metadata must have the same number of factors (cols), but not necessarily samples (row)
    if metadata_1.total_num_factors != metadata_2.total_num_factors:
        raise ValueError(
            f"Number of factors differs between metadata_1 ({metadata_1.total_num_factors}) "
            f"and metadata_2 ({metadata_2.total_num_factors})"
        )

    # Validate and attach discrete data
    if metadata_1.discrete_factor_names:
        _compare_keys(metadata_1.discrete_factor_names, metadata_2.discrete_factor_names)
        _validate_factors_and_data(metadata_1.discrete_factor_names, metadata_1.discrete_data)

        factor_names.extend(metadata_1.discrete_factor_names)
        m1_data.append(metadata_1.discrete_data)
        m2_data.append(metadata_2.discrete_data)

    # Validate and attach continuous data
    if metadata_1.continuous_factor_names:
        _compare_keys(metadata_1.continuous_factor_names, metadata_2.continuous_factor_names)
        _validate_factors_and_data(metadata_1.continuous_factor_names, metadata_1.continuous_data)

        factor_names.extend(metadata_1.continuous_factor_names)
        m1_data.append(metadata_1.continuous_data)
        m2_data.append(metadata_2.continuous_data)

    # Turns list of discrete and continuous into one array
    return factor_names, m1_data, m2_data


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


def most_deviated_factors(
    metadata_1: Metadata,
    metadata_2: Metadata,
    ood: OODOutput,
) -> list[tuple[str, float]]:
    """
    Determines greatest deviation in metadata features per out of distribution sample in metadata_2.

    Parameters
    ----------
    metadata_1 : Metadata
        A reference set of Metadata containing factor names and samples
        with discrete and/or continuous values per factor
    metadata_2 : Metadata
        The set of Metadata that is tested against the reference metadata.
        This set must have the same number of features but does not require the same number of samples.
    ood : OODOutput
        A class output by the DataEval's OOD functions that contains which examples are OOD.

    Returns
    -------
    list[tuple[str, float]]
        An array of the factor name and deviation of the highest metadata deviation for each OOD example in metadata_2.

    Notes
    -----
    1. Both :class:`.Metadata` inputs must have discrete and continuous data in the shape (samples, factors)
       and have equivalent factor names and lengths
    2. The flag at index `i` in :attr:`.OODOutput.is_ood` must correspond
       directly to sample `i` of `metadata_2` being out-of-distribution from `metadata_1`

    Examples
    --------

    >>> from dataeval.detectors.ood import OODOutput

    All samples are out-of-distribution

    >>> is_ood = OODOutput(np.array([True, True, True]), np.array([]), np.array([]))
    >>> most_deviated_factors(metadata1, metadata2, is_ood)
    [('time', 2.0), ('time', 2.592), ('time', 3.51)]

    If there are no out-of-distribution samples, a list is returned

    >>> is_ood = OODOutput(np.array([False, False, False]), np.array([]), np.array([]))
    >>> most_deviated_factors(metadata1, metadata2, is_ood)
    []
    """

    ood_mask: NDArray[np.bool] = ood.is_ood

    # No metadata correlated with out of distribution data
    if not any(ood_mask):
        return []

    # Combines reference and test factor names and data if exists and match exactly
    # shape -> (samples, factors)
    factor_names, md_1, md_2 = _combine_metadata(
        metadata_1=metadata_1,
        metadata_2=metadata_2,
    )

    # Stack discrete and continuous factors as separate factors. Must have equal sample counts
    metadata_ref = np.hstack(md_1) if md_1 else np.array([])
    metadata_tst = np.hstack(md_2) if md_2 else np.array([])

    if len(metadata_ref) < 3:
        warnings.warn(
            f"At least 3 reference metadata samples are needed, got {len(metadata_ref)}",
            UserWarning,
        )
        return []

    if len(metadata_tst) != len(ood_mask):
        raise ValueError(
            f"ood and test metadata must have the same length, "
            f"got {len(ood_mask)} and {len(metadata_tst)} respectively."
        )

    # Calculates deviations of all samples in m2_data
    # from the median values of the corresponding index in m1_data
    # Guaranteed for inputs to not be empty
    deviations = _calc_median_deviations(metadata_ref, metadata_tst)

    # Get most impactful factor deviation of each sample for ood samples only
    deviation = np.max(deviations, axis=1)[ood_mask].astype(np.float16)

    # Get indices of most impactful factors for ood samples only
    max_factors = np.argmax(deviations, axis=1)[ood_mask]

    # Get names of most impactful factors TODO: Find better way than np.dtype(<U4)
    most_ood_factors = np.array(factor_names)[max_factors].tolist()

    # List of tuples matching the factor name with its deviation

    return [(factor, dev) for factor, dev in zip(most_ood_factors, deviation)]
