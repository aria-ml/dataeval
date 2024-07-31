import numpy as np
from sklearn.metrics import average_precision_score


def uap(labels: np.ndarray, scores: np.ndarray):
    return float(average_precision_score(labels, scores, average="weighted"))
