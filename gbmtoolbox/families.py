"""JAX likelihood-family primitives for GBM Toolbox."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .linalg import logdet_spd, solve_spd


@dataclass(frozen=True)
class FamilyOps:
    name: str

    def mean(self, eta, observation_covariance=None):
        eta = jnp.atleast_1d(eta)
        if self.name == "gaussian":
            return eta
        if self.name == "bernoulli":
            return jax.nn.sigmoid(eta)
        if self.name == "categorical":
            return jax.nn.softmax(eta)
        raise ValueError(f"unsupported family {self.name!r}")

    def log_prob(self, y, eta, observation_covariance=None):
        eta = jnp.atleast_1d(eta)
        if self.name == "gaussian":
            yv = jnp.atleast_1d(y).astype(eta.dtype)
            delta = yv - eta
            R = observation_covariance
            d = eta.shape[0]
            quad = delta @ solve_spd(R, delta)
            return -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet_spd(R) + quad)
        if self.name == "bernoulli":
            yv = jnp.atleast_1d(y).astype(eta.dtype)
            return jnp.sum(yv * eta - jax.nn.softplus(eta))
        if self.name == "categorical":
            yi = jnp.asarray(y, dtype=jnp.int32).reshape(())
            return eta[yi] - logsumexp(eta)
        raise ValueError(f"unsupported family {self.name!r}")

    def score(self, y, eta, observation_covariance=None):
        eta = jnp.atleast_1d(eta)
        if self.name == "gaussian":
            yv = jnp.atleast_1d(y).astype(eta.dtype)
            return solve_spd(observation_covariance, yv - eta)
        if self.name == "bernoulli":
            yv = jnp.atleast_1d(y).astype(eta.dtype)
            return yv - jax.nn.sigmoid(eta)
        if self.name == "categorical":
            yi = jnp.asarray(y, dtype=jnp.int32).reshape(())
            p = jax.nn.softmax(eta)
            return jax.nn.one_hot(yi, eta.shape[0], dtype=eta.dtype) - p
        raise ValueError(f"unsupported family {self.name!r}")

    def fisher(self, eta, observation_covariance=None):
        eta = jnp.atleast_1d(eta)
        if self.name == "gaussian":
            eye = jnp.eye(eta.shape[0], dtype=eta.dtype)
            return solve_spd(observation_covariance, eye)
        if self.name == "bernoulli":
            p = jax.nn.sigmoid(eta)
            return jnp.diag(p * (1.0 - p))
        if self.name == "categorical":
            p = jax.nn.softmax(eta)
            return jnp.diag(p) - jnp.outer(p, p)
        raise ValueError(f"unsupported family {self.name!r}")

    def sample(self, key, eta, observation_covariance=None):
        eta = jnp.atleast_1d(eta)
        if self.name == "gaussian":
            return jax.random.multivariate_normal(key, eta, observation_covariance)
        if self.name == "bernoulli":
            draw = jax.random.bernoulli(key, logits=eta)
            return draw.astype(jnp.int32)
        if self.name == "categorical":
            return jax.random.categorical(key, eta)
        raise ValueError(f"unsupported family {self.name!r}")


def get_family(name: str) -> FamilyOps:
    family = str(name).lower()
    if family not in {"gaussian", "bernoulli", "categorical"}:
        raise ValueError("family must be 'gaussian', 'bernoulli', or 'categorical'")
    return FamilyOps(family)
