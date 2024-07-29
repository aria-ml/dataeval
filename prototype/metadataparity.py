import warnings

import numpy as np
import scipy
from metadata import str2int, validate_dict


class MetadataParity:
    def __init__(self):
        self.factors = None

    def set_factors(self, factors, continuous_factor_names=None, continuous_factor_bincounts=None):
        """
        Sets up the internal list of metadata factors.

        Parameters
        ----------
        factors : Dict
            Each key of 'factors' should be a metadata factor, where the value
            is a list, where factors[key][i] is a string representing the
            value of the metadata factor key at the ith element of the dataset.
            The values for the 'class' factor are ignored.
        continuous_factor_names : list[str], default=None
            Factors whose values span a continuous range should be handled
            differently from the default behavior, which is to treat factors
            as discrete. This is a list of all factor names that should
            be treated as continuous values.
        """
        # self.df = df[df.country_code == "RUS"]
        if continuous_factor_names is None:
            continuous_factor_names= []
        if continuous_factor_bincounts is None:
            continuous_factor_bincounts = [10] * len(continuous_factor_names)
        
        if len(continuous_factor_bincounts) != len(continuous_factor_names):
            raise Exception(f"continuous_factor_bincounts has length {len(continuous_factor_bincounts)}, "
                            f"but continuous_factor_names has length {len(continuous_factor_names)}. "
                            "Each element of continuous_factor_names must have a corresponding elemment "
                            "in continuous_factor_bincounts. You can also leave continuous_factor_bincounts empty "
                            "to use a default quantization of 10 bins.")
        
        self.factors = factors.copy()

        # make sure each factor has the same number of entries
        validate_dict(self.factors)
        
        #self.factors = str2int(self.factors)

        self.all_factor_names = list(self.factors.keys())[1:]

        self.continuous_factor_names = continuous_factor_names

        self.discrete_factor_names = []
        for factor_name in self.all_factor_names:
            if factor_name not in self.continuous_factor_names:
                self.discrete_factor_names.append(factor_name)
        
        self.num_bins = {}
        for i, continuous_factor in enumerate(continuous_factor_names):
            self.num_bins[continuous_factor] = continuous_factor_bincounts[i]

    def evaluate(self):
        """
        Evaluates the statistical indepedence of metadata factors from class labels.
        This performs a chi-square test, which provides a score and a p-value for
        statistical independence between each pair of a metadata factor and a class label.
        A high score with a low p-value suggests that a metadata factor is strongly
        correlated with a class label.

        Returns
        -------
        np.ndarray
            Matrix (num_factors) whose (i)th element corresponds to
            the chi-square score for the relationship between factor i
            and the class labels in the dataset.
        nd.ndarray
            Matrix (num_factors) whose (i)th element corresponds to
            the p-value value for the chi-square test for the relationship between
            factor i and the class labels in the dataset.
        """
        if self.factors is None:
            raise Exception("Must call set_factors(factors) before calling evaluate()")

        chi_matrix = [0] * len(self.all_factor_names)
        p_matrix = [0] * len(self.all_factor_names)
        labels = self.factors["class"]
        n_cls = len(np.unique(labels))
        for i, current_factor_name in enumerate(self.all_factor_names):
            if current_factor_name in self.continuous_factor_names:
                # If current_factor_name is a continuous factor, this
                # quantizes it so that it can be treated as a discrete factor
                try:
                    hist, bin_edges = np.histogram(self.factors[current_factor_name], bins=self.num_bins[current_factor_name])
                except TypeError:
                    raise TypeError(f"Encountered a non-numeric value for factor {current_factor_name}, but the factor"
                                    " was specified to be continuous. Try ensuring all occurrences of this factor are numeric types,"
                                    f" or remove {current_factor_name} from the continuous_factor_names argument when calling set_factors.")
                # This subverts a problem with np.histogram and np.digitize where a dataset could be quantized into more bins than were specified
                # # by self.num_bins when data points lay on bin edges.
                # For example, if [0,0,1,1] was histogrammed into one bin, this ensures that only one bin is represented when the data is digitized.
                bin_edges[-1] = np.inf
                bin_edges[0] = -np.inf

                digitized = np.digitize(self.factors[current_factor_name], bin_edges)

                # List of the quantized values of this factor at each point in the dataset
                factor_values = digitized
            else:
                # List of the values of this factor at each point in the dataset
                factor_values = self.factors[current_factor_name]

            unique_factor_values = np.unique(factor_values)
            contingency_matrix = np.zeros((len(unique_factor_values), n_cls))
            # Builds a contingency matrix where entry at index (r,c) represents
            # the frequency of current_factor_name achieving value unique_factor_values[r]
            # at a data point with class c.
            for label in range(n_cls):
                for fi, factor_value in enumerate(unique_factor_values):
                    with_this_label = np.where(labels == label)[0]
                    with_this_factor_value = np.where(factor_values == factor_value)[0]
                    with_both = np.intersect1d(with_this_label, with_this_factor_value)
                    contingency_matrix[fi, label] = len(with_both)
                    if 0 < contingency_matrix[fi, label] < 5:
                        warnings.warn(
                            f"Factor {current_factor_name} value {factor_value} co-occurs "
                            f"only {contingency_matrix[fi, label]} times with label {label}. "
                            "This can cause inaccurate chi_square calculation. Recommend"
                            "ensuring each label occurs either 0 times or at least 5 times. "
                            "If you are using continuous factors, try quantizing its values "
                            "into fewer bins."
                        )

            # Trims factor values with zero occurrences from the contingency matrix,
            # since scipy.stats.chi2_contingency fails when there are zero rows.
            rowmask = np.array([True] * np.shape(contingency_matrix)[0])
            for row in range(len(rowmask)):
                if np.sum(contingency_matrix[row, :]) == 0:
                    rowmask[row] = False
            contingency_matrix = contingency_matrix[rowmask,:]

            chi2, p, dof, ef = scipy.stats.chi2_contingency(contingency_matrix)

            chi_matrix[i] = chi2
            p_matrix[i] = p
        return chi_matrix, p_matrix
