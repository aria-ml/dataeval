"""build_model_input must accept MAITE (image, target, metadata) tuples, not only bare images.

Passing a full MAITE dataset batch (tuples) to a predictor's __call__ previously crashed
because the tuple itself was treated as the image.
"""

import numpy as np

from dataeval.models import ModelIOSpec, build_model_input


def _spec():
    return ModelIOSpec(
        task="IMAGE_CLASSIFICATION",
        channels="RGB",
        height=8,
        width=8,
        batch_size=-1,
        n_classes=4,
    )


def test_accepts_maite_tuples_like_bare_images():
    imgs = [np.zeros((3, 16, 16), dtype=np.uint8), np.full((3, 12, 20), 255, dtype=np.uint8)]
    # MAITE-style datum tuples: (image, target, metadata)
    tuples = [(imgs[0], np.array([1, 0, 0, 0]), {"id": 0}), (imgs[1], np.array([0, 1, 0, 0]), {"id": 1})]

    from_bare = build_model_input(imgs, _spec())
    from_tuples = build_model_input(tuples, _spec())

    assert from_tuples.shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(from_bare, from_tuples)
