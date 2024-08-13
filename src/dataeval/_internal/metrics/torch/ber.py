from typing import Dict

import torch

from dataeval._internal.metrics import ber


class BER(ber.BER):
    def __init__(self, method: ber._METHODS = "KNN", k: int = 1) -> None:
        super().__init__(method=method, k=k)

    def evaluate(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculates the Bayes Error Rate estimate using the provided method

        Parameters
        ----------
        images : torch.Tensor (N, : )
            Array of images or image embeddings
        labels : torch.Tensor (N, 1)
            Array of labels for each image or image embedding

        Returns
        -------
        Dict[str, float]
            ber : float
                The estimated lower bounds of the Bayes Error Rate
            ber_lower : float
                The estimated upper bounds of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        _images = images.detach().cpu().numpy()  # Converts `torch.Tensor`` to `numpy.ndarray``
        _labels = labels.detach().cpu().numpy()
        return super().evaluate(_images, _labels)
