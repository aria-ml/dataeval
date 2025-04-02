from __future__ import annotations

__all__ = []

from typing import NamedTuple

from dataeval.outputs._base import MappingOutput, SequenceOutput


class MostDeviatedFactorsOutput(SequenceOutput[tuple[str, float]]):
    """
    Output class for results of :func:`.most_deviated_factors` for OOD samples with metadata.

    Attributes
    ----------
    value : tuple[str, float]
        A tuple of the factor name and deviation of the highest metadata deviation
    """


class MetadataDistanceValues(NamedTuple):
    """
    Statistics comparing metadata distance.

    Attributes
    ----------
    statistic : float
        the KS statistic
    location : float
        The value at which the KS statistic has its maximum, measured in IQR-normalized units relative
        to the median of the reference distribution.
    dist : float
        The Earth Mover's Distance normalized by the interquartile range (IQR) of the reference
    pvalue : float
        The p-value from the KS two-sample test
    """

    statistic: float
    location: float
    dist: float
    pvalue: float


class MetadataDistanceOutput(MappingOutput[str, MetadataDistanceValues]):
    """
    Output class for results of ks_2samp featurewise comparisons of new metadata to reference metadata.

    Attributes
    ----------
    key : str
        Metadata feature names
    value : :class:`.MetadataDistanceValues`
        Output per feature name containing the statistic, statistic location, distance, and pvalue.
    """


class OODPredictorOutput(MappingOutput[str, float]):
    """
    Output class for results of :func:`find_ood_predictors` for the
    mutual information between factors and being out of distribution
    """
