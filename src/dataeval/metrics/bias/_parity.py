from __future__ import annotations

import warnings

__all__ = []


from dataeval.core._label_parity import label_parity as _label_parity
from dataeval.core._parity import parity as _parity
from dataeval.data import Metadata
from dataeval.outputs import LabelParityOutput, ParityOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike


@set_metadata
def label_parity(
    expected_labels: ArrayLike,
    observed_labels: ArrayLike,
    num_classes: int | None = None,
) -> LabelParityOutput:
    """
    Calculate the chi-square statistic to assess the :term:`parity<Parity>` \
    between expected and observed label distributions.

    This function computes the frequency distribution of classes in both expected and observed labels, normalizes
    the expected distribution to match the total number of observed labels, and then calculates the chi-square
    statistic to determine if there is a significant difference between the two distributions.

    Parameters
    ----------
    expected_labels : ArrayLike
        List of class labels in the expected dataset
    observed_labels : ArrayLike
        List of class labels in the observed dataset
    num_classes : int or None, default None
        The number of unique classes in the datasets. If not provided, the function will infer it
        from the set of unique labels in expected_labels and observed_labels

    Returns
    -------
    LabelParityOutput
        chi-squared score and :term`P-Value` of the test

    Raises
    ------
    ValueError
        If expected label distribution is empty, is all zeros, or if there is a mismatch in the number
        of unique classes between the observed and expected distributions.


    Note
    ----
    - Providing ``num_classes`` can be helpful if there are classes with zero instances in one of the distributions.
    - The function first validates the observed distribution and normalizes the expected distribution so that it
      has the same total number of labels as the observed distribution.
    - It then performs a :term:`Chi-Square Test of Independence` to determine if there is a statistically significant
      difference between the observed and expected label distributions.
    - This function acts as an interface to the scipy.stats.chisquare method, which is documented at
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html


    Examples
    --------
    Randomly creating some label distributions using ``np.random.default_rng``

    >>> rng = np.random.default_rng(175)
    >>> expected_labels = rng.choice([0, 1, 2, 3, 4], (100))
    >>> observed_labels = rng.choice([2, 3, 0, 4, 1], (100))
    >>> label_parity(expected_labels, observed_labels)
    LabelParityOutput(score=14.007374204742625, p_value=0.0072715574616218)
    """
    return LabelParityOutput(*_label_parity(expected_labels, observed_labels, num_classes=num_classes))


@set_metadata
def parity(metadata: Metadata) -> ParityOutput:
    """
    Calculate chi-square statistics to assess the linear relationship \
    between multiple factors and class labels.

    This function computes the chi-square statistic for each metadata factor to determine if there is
    a significant relationship between the factor values and class labels. The chi-square statistic is
    only valid for linear relationships. If non-linear relationships exist, use `balance`.

    Parameters
    ----------
    metadata : Metadata
        Preprocessed metadata

    Returns
    -------
    ParityOutput[NDArray[np.float64]]
        Arrays of length (num_factors) whose (i)th element corresponds to the
        chi-square score and :term:`p-value<P-Value>` for the relationship between factor i and
        the class labels in the dataset.

    Raises
    ------
    Warning
        If any cell in the contingency matrix has a value between 0 and 5, a warning is issued because this can
        lead to inaccurate chi-square calculations. It is recommended to ensure that each label co-occurs with
        factor values either 0 times or at least 5 times.

    Note
    ----
    - A high score with a low p-value suggests that a metadata factor is strongly correlated with a class label.
    - The function creates a contingency matrix for each factor, where each entry represents the frequency of a
      specific factor value co-occurring with a particular class label.
    - Rows containing only zeros in the contingency matrix are removed before performing the chi-square test
      to prevent errors in the calculation.

    See Also
    --------
    balance

    Examples
    --------
    Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

    >>> metadata = generate_random_metadata(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={
    ...         "age": [25, 30, 35, 45],
    ...         "income": [50000, 65000, 80000],
    ...         "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175)

    >>> parity(metadata)
    ParityOutput(score=array([7.357, 5.467, 0.515]), p_value=array([0.289, 0.243, 0.773]), factor_names=['age', 'income', 'gender'], insufficient_data={'age': {35: {'artist': 4}, 45: {'artist': 4, 'teacher': 3}}, 'income': {50000: {'artist': 3}}})
    """  # noqa: E501
    factor_names = metadata.factor_names
    index2label = metadata.index2label

    if not factor_names:
        raise ValueError("No factors found in provided metadata.")

    output = _parity(metadata.binned_data, metadata.class_labels.tolist(), return_insufficient_data=True)

    insufficient_data = {
        factor_names[k]: {vk: {index2label[vvk]: vvv for vvk, vvv in vv.items()} for vk, vv in v.items()}
        for k, v in output[2].items()
    }

    if insufficient_data:
        warnings.warn(
            f"Factors {list(insufficient_data)} did not meet the recommended "
            "5 occurrences for each value-label combination."
        )

    return ParityOutput(
        score=output[0],
        p_value=output[1],
        factor_names=metadata.factor_names,
        insufficient_data=insufficient_data,
    )
