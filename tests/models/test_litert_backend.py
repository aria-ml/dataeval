"""Tests for the LiteRT backend (optional runtime)."""

from pathlib import Path

import numpy as np
import pytest

from dataeval.models._backends import LiteRtBackend, make_backend

tf = pytest.importorskip("tensorflow")  # used to build a tiny .tflite fixture


def _tiny_tflite(tmp_path: Path) -> Path:
    # A model that takes NHWC (1,8,8,3) and returns (1,4) scores via global mean + dense.
    inputs = tf.keras.Input(shape=(8, 8, 3), batch_size=1)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    out = tf.keras.layers.Dense(4, activation="softmax", name="scores")(x)
    model = tf.keras.Model(inputs, out)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    path = tmp_path / "classifier.tflite"
    path.write_bytes(conv.convert())
    return path


def test_litert_backend_runs_nchw_input(tmp_path: Path):
    path = _tiny_tflite(tmp_path)
    backend = make_backend(path)
    assert isinstance(backend, LiteRtBackend)
    out = backend.run(np.zeros((1, 3, 8, 8), dtype=np.float32))  # NCHW in
    name = backend.output_names[0]
    assert out[name].shape == (1, 4)
