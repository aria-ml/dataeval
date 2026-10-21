import numpy as np
import pytest

from dataeval.core import dhash, image_stats, phash, xxhash
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required


def _near_black() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 2, (3, 64, 64)).astype(np.uint8)


def test_near_black_uint8_perceptual_hashes_match_the_native_view():
    frame = _near_black()
    got = image_stats([frame], stats=ImageStats.HASH_PHASH | ImageStats.HASH_DHASH)["stats"]
    assert got["phash"][0] == phash(frame)
    assert got["dhash"][0] == dhash(frame)


def test_xxhash_still_digests_the_float64_anchor():
    frame = _near_black()
    got = image_stats([frame], stats=ImageStats.HASH_XXHASH)["stats"]["xxhash"][0]
    assert got == xxhash(frame.astype(np.float64))


def test_ordinary_frames_hash_as_before():
    frame = np.random.default_rng(1).integers(0, 256, (3, 64, 64)).astype(np.uint8)
    got = image_stats([frame], stats=ImageStats.HASH_PHASH)["stats"]["phash"][0]
    assert got == phash(frame) == phash(frame.astype(np.float64))
