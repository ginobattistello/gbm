"""Small JAX linear-algebra helpers used by filtering and inference."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def symmetrize(a):
    return 0.5 * (a + jnp.swapaxes(a, -1, -2))


def solve_spd(a, b, *, jitter: float = 0.0):
    """Solve ``a x = b`` for a symmetric positive-definite matrix."""
    a = symmetrize(a)
    if jitter:
        a = a + jitter * jnp.eye(a.shape[-1], dtype=a.dtype)
    L = jnp.linalg.cholesky(a)
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y, lower=False)


def logdet_spd(a, *, jitter: float = 0.0):
    a = symmetrize(a)
    if jitter:
        a = a + jitter * jnp.eye(a.shape[-1], dtype=a.dtype)
    L = jnp.linalg.cholesky(a)
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def psd_factor(a, *, tol: float = 1e-12):
    """Fixed-shape factor ``L`` such that ``L L.T`` equals a PSD matrix.

    Zero-eigenvalue directions are retained as zero columns.  This avoids the
    dynamic-rank slicing that is incompatible with JIT/scan.
    """
    a = symmetrize(a)
    vals, vecs = jnp.linalg.eigh(a)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(vals)))
    threshold = tol * scale
    active = jax.lax.stop_gradient(vals > threshold)
    roots = jnp.where(active, jnp.sqrt(jnp.maximum(vals, 0.0)), 0.0)
    return vecs * roots[None, :]
