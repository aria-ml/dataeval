"""
Out-of-distribution (OOD) detectors identify data that is different from the data used to train a particular model.
"""

__all__ = ["OODOutput", "OODScoreOutput", "OOD_AE", "OOD_KNN"]

from dataeval.detectors.ood.ae import OOD_AE
from dataeval.detectors.ood.knn import OOD_KNN
from dataeval.outputs._ood import OODOutput, OODScoreOutput
