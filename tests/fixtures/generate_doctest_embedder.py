"""Regenerate the frozen embedding-model weights used by the doctest fixtures.

The doctest embedding model (see ``_build_embedder`` / ``_create_model`` in
``src/conftest.py``) is a tiny untrained network. Its random initialization differs
every process, which makes any embedding-derived doctest output (e.g. the per-class
``dispersion`` reported by :class:`~dataeval.scope.Coverage`) non-reproducible. To let
those doctests show real, verified output, we freeze the weights once and check them in.

Run from the repo root to regenerate (only needed if the architecture changes)::

    python tests/fixtures/generate_doctest_embedder.py
"""

from pathlib import Path

import torch

WEIGHTS_PATH = Path(__file__).parent / "doctest_embedder.pt"


def build_embedder() -> torch.nn.Module:
    """Build the doctest embedding architecture with materialized (lazy) parameters."""
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((4, 4)),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )
    # Run a dummy forward so the LazyLinear materializes to a concrete shape.
    model(torch.randn(1, 3, 64, 64))
    return model


def main() -> None:
    """Initialize the embedder deterministically and write its weights to disk."""
    torch.manual_seed(0)
    model = build_embedder()
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Wrote frozen doctest embedder weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
