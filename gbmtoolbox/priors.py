"""Gaussian prior containers used by GBM Toolbox."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _as_mean(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("prior mean must be a non-empty finite vector")
    return arr


def covariance_matrix(value, dim: int, *, name: str, allow_semidefinite: bool = True) -> np.ndarray:
    """Expand scalar/vector/matrix covariance notation to a square matrix."""
    if value is None:
        raise ValueError(f"{name} cannot be None")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        cov = np.eye(dim) * float(arr)
    elif arr.ndim == 1:
        if arr.shape != (dim,):
            raise ValueError(f"{name} vector must have length {dim}")
        cov = np.diag(arr)
    elif arr.ndim == 2:
        if arr.shape != (dim, dim):
            raise ValueError(f"{name} matrix must have shape ({dim}, {dim})")
        cov = arr.copy()
    else:
        raise ValueError(f"{name} must be a scalar, vector, or square matrix")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    tol = 1e-12 * max(1.0, float(np.max(np.abs(eig))) if eig.size else 1.0)
    if allow_semidefinite:
        if np.min(eig) < -tol:
            raise ValueError(f"{name} must be positive semidefinite")
    elif np.min(eig) <= tol:
        raise ValueError(f"{name} must be positive definite")
    return 0.5 * (cov + cov.T)


@dataclass(frozen=True)
class GaussianPrior:
    """Multivariate Gaussian prior.

    ``covariance`` may be a scalar, diagonal vector, or full matrix. A zero
    marginal variance fixes that parameter at its prior mean. With a full
    covariance matrix, a fixed component must also have zero cross-covariance.
    """

    mean: Sequence[float]
    covariance: object
    names: Sequence[str] | None = None

    def __post_init__(self):
        mean = _as_mean(self.mean)
        raw = np.asarray(self.covariance, dtype=float)
        if raw.ndim == 2 and raw.shape == (len(mean), len(mean)):
            raw_diag = np.diag(raw)
            for j in np.flatnonzero(np.isclose(raw_diag, 0.0, atol=1e-14, rtol=0.0)):
                row = raw[j].copy()
                row[j] = 0.0
                if not np.allclose(row, 0.0, atol=1e-12, rtol=0.0):
                    raise ValueError("a zero-variance fixed parameter must have zero cross-covariance")
        cov = covariance_matrix(self.covariance, len(mean), name="prior covariance")
        diag = np.diag(cov)
        fixed = np.isclose(diag, 0.0, atol=1e-14, rtol=0.0)
        for j in np.flatnonzero(fixed):
            row = cov[j].copy()
            row[j] = 0.0
            if not np.allclose(row, 0.0, atol=1e-12, rtol=0.0):
                raise ValueError("a zero-variance fixed parameter must have zero cross-covariance")
        if self.names is None:
            names = None
        else:
            names = tuple(str(x) for x in self.names)
            if len(names) != len(mean):
                raise ValueError("names must contain one entry per prior parameter")
            if len(set(names)) != len(names):
                raise ValueError("prior parameter names must be unique")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "names", names)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    @property
    def fixed_mask(self) -> np.ndarray:
        return np.isclose(np.diag(self.covariance), 0.0, atol=1e-14, rtol=0.0)


def broadcast_prior(prior: GaussianPrior, dim: int, *, prefix: str) -> GaussianPrior:
    """Broadcast a one-dimensional prior independently over ``dim`` entries."""
    if dim < 1:
        raise ValueError("broadcast dimension must be >= 1")
    if prior.dim == dim:
        if prior.names is None:
            return GaussianPrior(prior.mean, prior.covariance, names=[f"{prefix}[{i}]" for i in range(dim)])
        return prior
    if prior.dim != 1:
        raise ValueError(f"{prefix} prior must have dimension 1 or {dim}")
    mean = np.repeat(prior.mean[0], dim)
    variance = float(prior.covariance[0, 0])
    cov = np.eye(dim) * variance
    if prior.names is None:
        names = [f"{prefix}[{i}]" for i in range(dim)]
    else:
        base = prior.names[0]
        names = [f"{base}[{i}]" for i in range(dim)]
    return GaussianPrior(mean, cov, names=names)


@dataclass(frozen=True)
class Priors:
    """Separate priors for static evolution and observation parameters."""

    evolution: GaussianPrior
    observation: GaussianPrior

    @property
    def dim(self) -> int:
        return self.evolution.dim + self.observation.dim

    @property
    def mean(self) -> np.ndarray:
        return np.concatenate([self.evolution.mean, self.observation.mean])

    @property
    def covariance(self) -> np.ndarray:
        a = self.evolution.covariance
        b = self.observation.covariance
        out = np.zeros((self.dim, self.dim), dtype=float)
        out[: a.shape[0], : a.shape[1]] = a
        out[a.shape[0] :, a.shape[1] :] = b
        # NOTE: block diag matrix assume prior independence between evolution and observation quantities
        return out

    @property
    def names(self) -> tuple[str, ...]:
        en = self.evolution.names or tuple(f"theta[{i}]" for i in range(self.evolution.dim))
        on = self.observation.names or tuple(f"phi[{i}]" for i in range(self.observation.dim))
        names = tuple(en) + tuple(on)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique across evolution and observation blocks")
        return names

    @property
    def theta_slice(self) -> slice:
        return slice(0, self.evolution.dim)

    @property
    def phi_slice(self) -> slice:
        return slice(self.evolution.dim, self.dim)
