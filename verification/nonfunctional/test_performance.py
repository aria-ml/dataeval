"""Verify memory scaling and timing of reference workloads.

Maps to meta repo test cases:
  - TC-25.1: Performance and scale

Marked ``performance`` so slow timing checks can be deselected with ``-m "not performance"``.
"""

from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pytest

import dataeval.config as config

pytestmark = pytest.mark.performance

N_REFERENCE = 10_000

# Time budgets in seconds for the 10,000-image reference workloads: twice the v1.1.4 baselines
# measured on the CI runner (Duplicates 7.96 s, Outliers 13.87 s, Balance 0.72 s, DriftKNeighbors
# 2.08 s; the development machine measured 2.8, 3.8, 0.5, and 2.6 s). Regression guards, not
# performance promises.
TIME_BUDGETS_S = {
    "Duplicates": 16.0,
    "Outliers": 28.0,
    "Balance": 1.5,
    "DriftKNeighbors": 4.5,
}


class LazyImages:
    """Image dataset that generates each image on access from a per-index seeded rng (no stored pixels)."""

    def __init__(self, n: int, shape: tuple[int, ...] = (3, 32, 32), seed: int = 0):
        self.n, self.shape, self.seed = n, shape, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> np.ndarray:
        return np.random.default_rng((self.seed, index)).random(self.shape, dtype=np.float32)


class PooledExtractor:
    """Average-pools 3x72x96 images by 3 to a 2304-float (about 9 KB) embedding."""

    def __call__(self, images) -> np.ndarray:
        batch = np.stack([np.asarray(image) for image in images])
        n = len(batch)
        return batch.reshape(n, 3, 24, 3, 32, 3).mean(axis=(3, 5)).reshape(n, -1)


@pytest.fixture(autouse=True)
def _batch_size():
    config.set_batch_size(64)
    yield
    config.set_batch_size(None)
    config.set_max_processes(None)


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


class TestPerformance:
    """Verify bounded memory, bounded workers, and reference-workload timings."""

    def test_cached_embeddings_memory_bounded(self, tmp_path):
        """Peak memory follows the stored embeddings, not the raw dataset, as the dataset grows."""
        from dataeval import Embeddings

        shape = (3, 72, 96)
        sizes = (500, 2000)
        embedding_bytes = 2304 * 4
        peaks = {}
        for n in sizes:
            embeddings = Embeddings(
                LazyImages(n, shape),
                extractor=PooledExtractor(),
                batch_size=32,
                path=tmp_path / f"emb-{n}.npy",
                memory_threshold=0.0,  # always use the on-disk (memory-mapped) cache
            )
            tracemalloc.start()
            try:
                embeddings.compute()
                peaks[n] = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            assert isinstance(embeddings._embeddings, np.memmap)
            assert embeddings[:].shape == (n, 2304)

        stored_growth = (sizes[1] - sizes[0]) * embedding_bytes
        raw_bytes = sizes[1] * int(np.prod(shape)) * 4
        print(f"peak bytes by dataset size: {peaks}; stored embeddings {sizes[1] * embedding_bytes}; raw {raw_bytes}")
        # Peak growth is at most ~3x the growth in stored embeddings, and far below the raw data.
        assert peaks[sizes[1]] - peaks[sizes[0]] <= 3 * stored_growth
        assert peaks[sizes[1]] < raw_bytes / 4

    def test_max_processes_bounds_workers(self, monkeypatch):
        """The worker pool never exceeds ``max_processes``; timings per setting are recorded."""
        import dataeval.core._compute_stats as compute_stats_module
        from dataeval.quality import Duplicates, Outliers

        observed: list[int] = []
        original = compute_stats_module.PoolWrapper

        class RecordingPool(original):
            def __init__(self, processes, *args, **kwargs):
                super().__init__(processes, *args, **kwargs)
                observed.append(getattr(self._pool, "_processes", 1) if self._pool is not None else 1)

        monkeypatch.setattr(compute_stats_module, "PoolWrapper", RecordingPool)

        dataset = LazyImages(2000)
        timings: dict[tuple[str, int], float] = {}
        for limit in sorted({1, 2, 4, os.cpu_count() or 1}):
            config.set_max_processes(limit)
            for name, evaluator in (("Duplicates", Duplicates()), ("Outliers", Outliers())):
                observed.clear()
                timings[(name, limit)] = _timed(lambda e=evaluator: e.evaluate(dataset))
                assert observed, f"{name} did not create a worker pool"
                assert max(observed) <= limit
        for (name, limit), seconds in timings.items():
            print(f"{name} max_processes={limit}: {seconds:.2f} s")
        assert len(timings) >= 2 * 2

    def test_reference_workloads_within_time_budgets(self):
        """Duplicates, Outliers, Balance, and DriftKNeighbors each finish inside their recorded budget."""
        from dataeval import Metadata
        from dataeval.bias import Balance
        from dataeval.quality import Duplicates, Outliers
        from dataeval.shift import DriftKNeighbors

        dataset = LazyImages(N_REFERENCE)
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 10, N_REFERENCE)
        factors = {f"continuous_{i}": rng.normal(size=N_REFERENCE) + 0.1 * labels for i in range(5)}
        factors |= {f"discrete_{i}": rng.integers(0, 5, N_REFERENCE) for i in range(5)}
        metadata = Metadata.from_factors(
            factors, labels, continuous_factor_bins={f"continuous_{i}": 10 for i in range(5)}
        )
        flat = np.stack([dataset[i].ravel() for i in range(N_REFERENCE)])
        reference, test = flat[:5000], flat[5000:]

        workloads = {
            "Duplicates": lambda: Duplicates().evaluate(dataset),
            "Outliers": lambda: Outliers().evaluate(dataset),
            "Balance": lambda: Balance().evaluate(metadata),
            "DriftKNeighbors": lambda: DriftKNeighbors().fit(reference).predict(test),
        }
        elapsed = {name: _timed(run) for name, run in workloads.items()}
        for name, seconds in elapsed.items():
            print(f"{name}: {seconds:.2f} s (budget {TIME_BUDGETS_S[name]:.1f} s)")
        over = {name: s for name, s in elapsed.items() if s > TIME_BUDGETS_S[name]}
        assert not over, f"over budget (seconds): {over}"
