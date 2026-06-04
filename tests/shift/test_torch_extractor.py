"""Tests for TorchExtractor batching and detection decoding."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval.extractors import TorchExtractor


class Linear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.required
class TestTorchExtractorBatching:
    def test_minibatching_matches_single_batch(self):
        """Same numeric result whether processed in one batch or several."""
        torch.manual_seed(0)
        model = Linear(6, 4)
        data = np.random.randn(10, 6).astype(np.float32)

        whole = TorchExtractor(model, device="cpu", batch_size=10)(data)
        chunked = TorchExtractor(model, device="cpu", batch_size=3)(data)

        assert whole.shape == (10, 4)
        assert chunked.shape == (10, 4)
        np.testing.assert_allclose(whole, chunked, rtol=1e-5, atol=1e-6)

    def test_none_batch_size_processes_all_without_global(self):
        """batch_size=None runs a single pass and needs no global batch size."""
        import dataeval.config as config

        original = config._config.batch_size
        config.set_batch_size(None)
        try:
            torch.manual_seed(0)
            model = Linear(6, 4)
            data = np.random.randn(7, 6).astype(np.float32)
            out = TorchExtractor(model, device="cpu")(data)  # no batch_size, no global
            assert out.shape == (7, 4)
            chunked = TorchExtractor(model, device="cpu", batch_size=2)(data)
            np.testing.assert_allclose(out, chunked, rtol=1e-5, atol=1e-6)
        finally:
            config.set_batch_size(original)

    def test_batch_size_property(self):
        model = Linear(6, 4)
        assert TorchExtractor(model, device="cpu", batch_size=8).batch_size == 8
        assert TorchExtractor(model, device="cpu").batch_size is None

    def test_empty_data(self):
        model = Linear(6, 4)
        out = TorchExtractor(model, device="cpu", batch_size=4)([])
        assert out.shape == (0,)

    def test_layer_hook_minibatching_matches_single_batch(self):
        """Layer-hook capture is correct across minibatches."""
        torch.manual_seed(0)

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(6, 5)
                self.b = nn.Linear(5, 4)

            def forward(self, x):
                return self.b(self.a(x))

        model = TwoLayer()
        data = np.random.randn(10, 6).astype(np.float32)

        whole = TorchExtractor(model, device="cpu", layer_name="a", batch_size=10)(data)
        chunked = TorchExtractor(model, device="cpu", layer_name="a", batch_size=3)(data)

        assert whole.shape == (10, 5)
        assert chunked.shape == (10, 5)
        np.testing.assert_allclose(whole, chunked, rtol=1e-5, atol=1e-6)


@pytest.mark.required
class TestTorchExtractorPostprocess:
    def test_postprocess_decodes_output(self):
        """postprocess_fn is applied per batch and concatenated."""
        model = Linear(6, 8)
        data = np.random.randn(5, 6).astype(np.float32)

        def decode(out):  # slice 8-wide output down to 3 class scores
            return out[:, :3]

        ex = TorchExtractor(model, device="cpu", batch_size=2, postprocess_fn=decode)
        out = ex(data)
        assert out.shape == (5, 3)

    def test_tuple_output_takes_first(self):
        """If the model returns a tuple, the first element is used."""

        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 4)

            def forward(self, x):
                y = self.fc(x)
                return y, x  # tuple output

        ex = TorchExtractor(TupleModel(), device="cpu", batch_size=4)
        out = ex(np.random.randn(3, 6).astype(np.float32))
        assert out.shape == (3, 4)

    def test_postprocess_receives_full_tuple_output(self):
        """postprocess_fn gets the FULL raw output (a tuple), not its element 0."""

        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 8)

            def forward(self, x):
                return self.fc(x), x  # (scores, aux) tuple, like a detection head

        def decode(output):
            scores, _aux = output  # must unpack the tuple -> fails if given only element 0
            return scores[:, :3]

        ex = TorchExtractor(TupleModel(), device="cpu", batch_size=2, postprocess_fn=decode)
        out = ex(np.random.randn(5, 6).astype(np.float32))
        assert out.shape == (5, 3)

    def test_postprocess_and_layer_name_conflict(self):
        with pytest.raises(ValueError, match="postprocess_fn.*layer_name"):
            TorchExtractor(Linear(6, 4), device="cpu", layer_name="fc", postprocess_fn=lambda o: o)
