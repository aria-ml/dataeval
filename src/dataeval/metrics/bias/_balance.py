from __future__ import annotations

__all__ = []


from dataeval.core import _balance
from dataeval.data import Metadata
from dataeval.outputs import BalanceOutput
from dataeval.outputs._base import set_metadata


@set_metadata
def balance(
    metadata: Metadata,
    num_neighbors: int = 5,
) -> BalanceOutput:
    """
    Mutual information (MI) between factors (class label, metadata, label/image properties).

    Parameters
    ----------
    metadata : Metadata
        Preprocessed metadata
    num_neighbors : int, default 5
        Number of points to consider as neighbors

    Returns
    -------
    BalanceOutput
        (num_factors+1) x (num_factors+1) estimate of mutual information \
            between num_factors metadata factors and class label. Symmetry is enforced.

    Note
    ----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> metadata = generate_random_metadata(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={
    ...         "age": [25, 30, 35, 45],
    ...         "income": [50000, 65000, 80000],
    ...         "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175)

    >>> bal = balance(metadata)
    >>> bal.balance
    array([1.017, 0.034, 0.   , 0.028])

    Return intra/interfactor balance (mutual information)

    >>> bal.factors
    array([[1.   , 0.015, 0.038],
           [0.015, 1.   , 0.008],
           [0.038, 0.008, 1.   ]])

    Return classwise balance (mutual information) of factors with individual class_labels

    >>> bal.classwise
    array([[7.818e-01, 1.388e-02, 1.803e-03, 7.282e-04],
           [7.084e-01, 2.934e-02, 1.744e-02, 3.996e-03],
           [7.295e-01, 1.157e-02, 2.799e-02, 9.451e-04]])


    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """
    if not metadata.factor_names:
        raise ValueError("No factors found in provided metadata.")

    factor_types = {k: v.factor_type for k, v in metadata.factor_info.items()}
    is_discrete = [factor_type != "continuous" for factor_type in factor_types.values()]

    nmi = _balance.balance(
        metadata.class_labels,
        metadata.binned_data,
        is_discrete,
        num_neighbors,
    )
    balance = nmi[0]
    factors = nmi[1:, 1:]

    classwise = _balance.balance_classwise(
        metadata.class_labels,
        metadata.binned_data,
        is_discrete,
        num_neighbors,
    )
    # Grabbing factor names for plotting function
    factor_names = ["class_label"] + list(metadata.factor_names)

    return BalanceOutput(balance, factors, classwise, factor_names, metadata.class_names)
