import warnings

import numpy as np
import scipy
from metadata import str2int, validate_dict
from dataeval._internal.functional.metadataparity import format_discretize_factors, compute_parity
from typing import Tuple, Optional, Any, List


class MetadataParity:
    def __init__(self, data_factors: dict[str, List[Any]], continuous_factor_names: Optional[np.ndarray] = None, continuous_factor_bincounts: Optional[np.ndarray] = None):
        """
        Sets up the internal list of metadata factors.

        Parameters
        ----------
        data_factors: Dict[str, List[Any]]
            The dataset factors, which are per-image attributes including class label and metadata.
            Each key of dataset_factors is a factor, whose value is the per-image factor values.
        continuous_factor_names : np.ndarray, default None
            The factors in data_factors that have continuous values.
            All factors are treated as having discrete values unless they
            are specified in this array. Each element of this array must occur as a key in data_factors.
        continuous_factor_bincounts : np.ndarray, default None
            Array of the bin counts to discretize values into for each factor in continuous_factor_names.
        """

        continuous_factor_names = np.array([], dtype=str) if continuous_factor_names is None else np.array(continuous_factor_names)
        continuous_factor_bincounts = 10 * np.ones(len(continuous_factor_names), dtype=int) if continuous_factor_bincounts is None else np.array(continuous_factor_bincounts)
        
        self.metadata_factors, self.labels = format_discretize_factors(data_factors, continuous_factor_names, continuous_factor_bincounts)

    def evaluate(self) -> dict[str,List[np.float64]]:
        """
        Evaluates the statistical indepedence of metadata factors from class labels.
        This performs a chi-square test, which provides a score and a p-value for
        statistical independence between each pair of a metadata factor and a class label.
        A high score with a low p-value suggests that a metadata factor is strongly
        correlated with a class label.

        Returns
        -------
        Dict[str, List[np.float64]]
            chi_square: np.ndarray
                Array of length (num_factors) whose (i)th element corresponds to
                the chi-square score for the relationship between factor i
                and the class labels in the dataset.
            p_values: np.ndarray
                Array of length (num_factors) whose (i)th element corresponds to
                the p-value for the chi-square test for the relationship between
                factor i and the class labels in the dataset.
        """
        chi_square, p_values = compute_parity(self.metadata_factors, self.labels)

        formatted_output = {
            'chi_squares': chi_square,
            'p_values': p_values
        }
        return formatted_output