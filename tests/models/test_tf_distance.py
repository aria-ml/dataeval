import numpy as np
import tensorflow as tf

from daml._internal.models.tensorflow.autoencoder import (
    relative_euclidean_distance,
)


def test_relative_euclidean_distance():
    x = tf.convert_to_tensor(np.random.rand(5, 3))
    y = tf.convert_to_tensor(np.random.rand(5, 3))

    assert (relative_euclidean_distance(x, y).numpy() == relative_euclidean_distance(y, x).numpy()).all()  # type: ignore
    assert (relative_euclidean_distance(x, x).numpy() == relative_euclidean_distance(y, y).numpy()).all()  # type: ignore
    assert (relative_euclidean_distance(x, y).numpy() >= 0.0).all()  # type: ignore
