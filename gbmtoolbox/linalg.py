"""Small JAX linear-algebra helpers used by filtering and inference."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def symmetrize(a):
    """Average a matrix with its transpose, removing round-off asymmetry."""
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
    """Log determinant of a symmetric positive-definite matrix, via Cholesky.

    Summing ``2*log(diag(L))`` avoids forming the determinant itself, which
    would overflow or underflow for the matrix sizes used here.
    """
    a = symmetrize(a)
    if jitter:
        a = a + jitter * jnp.eye(a.shape[-1], dtype=a.dtype)
    L = jnp.linalg.cholesky(a)
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
