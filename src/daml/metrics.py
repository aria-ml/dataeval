from typing import Dict, List


class Metrics:
    """
    Class aggregating daml Metrics by provider and function.
    """

    def __init__(self) -> None:
        self.outlier_detect_algos: Dict[str, List[str]] = {}
        self._get_outlier_detect_algos()

    def _get_outlier_detect_algos(self) -> Dict[str, List[str]]:
        self.outlier_detect_algos.update(
            {
                "alibi_detect": [
                    "Auto-Encoder (AE)",
                    "Variational Auto-Encoder (VAE)",
                    "Auto-Encoding Gaussian Mixture Model (AEGMM)",
                    "Variational AEGMM (VAEGMM)",
                    "Likelihood Ratios (LR)",
                ]
            }
        )
        return self.outlier_detect_algos

    def list_metrics(self) -> Dict[str, List[str]]:
        """
        Return all the supported outlier detection methods organized by
        provider.

        :return: a dictionary of { providers: [outlier_detection_methods] }.
        :rtype: Dict[str, List[str]]

        """
        return self.outlier_detect_algos
