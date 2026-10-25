"""Synthetic classification populations with known posteriors (isotropic Gaussians, sigma 1), and a Units builder."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.performance._feasibility_units import Units

F = NDArray[np.float64]


@dataclass(frozen=True)
class Population:
    name: str
    n_classes: int
    priors: F
    means: F  # (M, d)
    lossy: bool = False

    def posterior(self, X: F) -> F:
        # -0.5*||x-mu||^2 = x.mu - 0.5*||mu||^2 + (-0.5*||x||^2); the last term is constant per row and cancels
        logit = np.log(self.priors) + X @ self.means.T - 0.5 * (self.means**2).sum(axis=1)
        logit -= logit.max(axis=1, keepdims=True)
        p = np.exp(logit)
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[F, NDArray[np.intp], F]:
        comp = rng.choice(self.n_classes, size=n, p=self.priors)
        X = self.means[comp] + rng.standard_normal((n, self.means.shape[1]))
        post = self.posterior(X)
        # label from the posterior, not the component, so primary labels and replicas share one protocol
        return X, draw_labels(post, 1, rng)[:, 0], post

    def represent(self, X: F) -> F:
        return X[:, :2] if self.lossy else X


def _unit_means(m: int, d: int, scale: float, seed: int) -> F:
    u = np.random.default_rng(seed).standard_normal((m, d))
    return scale * u / np.linalg.norm(u, axis=1, keepdims=True)


def make_population(name: str, *, lossy: bool = False, seed: int = 0) -> Population:
    if name == "gauss2":  # Bayes accuracy Phi(1)
        means = np.zeros((2, 8))
        means[0, 0], means[1, 0] = 1.0, -1.0
        priors = np.array([0.5, 0.5])
    elif name == "gauss10":
        means, priors = _unit_means(10, 16, 3.0, seed), np.full(10, 0.1)
    elif name == "rare5":
        means, priors = _unit_means(5, 16, 3.0, seed), np.array([0.60, 0.20, 0.10, 0.07, 0.03])
    else:
        raise ValueError(f"unknown population {name!r}")
    return Population(name, len(priors), priors, means, lossy)


def draw_labels(post: F, m: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """M independent labels per row from the posterior (replicas under a correct protocol)."""
    cdf = post.cumsum(axis=1)
    u = rng.random((len(post), m, 1)) * cdf[:, None, -1:]
    return np.minimum((u > cdf[:, None, :]).sum(axis=2), post.shape[1] - 1).astype(np.intp)


def make_sequences(
    pop: Population, n_seq: int, *, min_len: int = 1, max_len: int = 100, rho: float = 0.8, rng: np.random.Generator
) -> tuple[F, NDArray[np.intp], F, NDArray[np.intp], NDArray[np.intp]]:
    """Dependent frames: rho*c_s + sqrt(1-rho^2)*eps around a per-sequence draw c_s. max_len is the declared bound."""
    lengths = rng.integers(min_len, max_len + 1, size=n_seq)
    seq_id = np.repeat(np.arange(n_seq), lengths)
    centers, _, _ = pop.sample(n_seq, rng)
    X = rho * centers[seq_id] + np.sqrt(1 - rho**2) * rng.standard_normal((len(seq_id), centers.shape[1]))
    post = pop.posterior(X)
    return X, draw_labels(post, 1, rng)[:, 0], post, seq_id.astype(np.intp), lengths.astype(np.intp)


def make_units(
    labels: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    n_classes: int | None = None,
    class_names: list[str] | None = None,
) -> Units:
    """Units over a fully labeled synthetic dataset: item i is dataset item i with datum id i."""
    y = np.asarray(labels, dtype=np.intp)
    n = len(y)
    codes = np.arange(n) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)
    return Units(
        labels=y,
        item_indices=np.arange(n, dtype=np.intp),
        groups=codes.astype(np.intp),
        n_classes=int(y.max()) + 1 if n_classes is None else n_classes,
        item_ids=np.arange(n),
        n_excluded=0,
        population="items of synthetic",
        class_names=class_names,
        grouped=bool(len(codes)) and int(np.bincount(codes).max()) > 1,
    )
