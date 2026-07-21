"""BoVWExtractor.transform / __call__ must accept unsized iterables (generators).

The docstring advertises "Iterable of images", but transform pre-allocated with
len(data), which crashes on a generator/iterator.
"""

import numpy as np
import pytest

from dataeval.extractors import BoVWExtractor


@pytest.fixture
def rgb_images():
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(5)]


@pytest.mark.optional
def test_transform_accepts_generator(rgb_images):
    extractor = BoVWExtractor(vocab_size=16)
    extractor.fit(rgb_images)

    from_list = np.asarray(extractor.transform(rgb_images))
    # a one-shot generator (no __len__) must work identically
    from_gen = np.asarray(extractor.transform(img for img in rgb_images))

    assert from_gen.shape == from_list.shape == (5, from_list.shape[1])
    np.testing.assert_array_equal(from_list, from_gen)


@pytest.mark.optional
def test_call_accepts_generator(rgb_images):
    extractor = BoVWExtractor(vocab_size=16)
    extractor.fit(rgb_images)
    out = np.asarray(extractor(img for img in rgb_images))
    assert out.shape[0] == 5
