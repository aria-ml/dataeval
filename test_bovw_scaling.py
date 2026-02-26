"""Quick benchmark: BoVW with 1 process vs 8 processes."""

import time

import numpy as np

from dataeval.config import set_max_processes
from dataeval.extractors import BoVWExtractor


def main() -> None:
    rng = np.random.default_rng(42)

    configs = [
        (200, 128),
        (500, 128),
        (500, 256),
    ]

    for num_images, img_size in configs:
        images = [rng.integers(0, 256, (3, img_size, img_size), dtype=np.uint8) for _ in range(num_images)]

        print(f"\nDataset: {num_images} images, {img_size}x{img_size} RGB")
        print("-" * 55)

        for n_procs in (1, 8):
            set_max_processes(n_procs)
            extractor = BoVWExtractor(vocab_size=256)

            t0 = time.perf_counter()
            embeddings = extractor(images)  # fit + transform
            elapsed = time.perf_counter() - t0

            print(f"  Processes: {n_procs:>2}  |  Time: {elapsed:6.2f}s  |  Shape: {embeddings.shape}")

    print()

    # Correctness check: fit+transform end-to-end with 1 vs 8 processes.
    # (fit now sorts by index so descriptor order is deterministic regardless
    #  of process count, giving identical KMeans clusters and histograms.)
    print("=" * 55)
    print("Correctness check: full fit+transform, 1 proc vs 8 proc")
    print("-" * 55)

    from dataeval.config import set_seed

    rng2 = np.random.default_rng(99)
    test_images = [rng2.integers(0, 256, (3, 128, 128), dtype=np.uint8) for _ in range(100)]

    set_seed(0)
    set_max_processes(1)
    result_1 = BoVWExtractor(vocab_size=128)(test_images)

    set_seed(0)
    set_max_processes(8)
    result_8 = BoVWExtractor(vocab_size=128)(test_images)

    match = np.allclose(result_1, result_8)
    max_diff = float(np.max(np.abs(np.asarray(result_1) - np.asarray(result_8))))
    print(f"  Shapes match:  {result_1.shape == result_8.shape}")
    print(f"  Values match:  {match}  (max abs diff: {max_diff:.2e})")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
