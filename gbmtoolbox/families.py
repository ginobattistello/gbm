"""JAX likelihood-family primitives for GBM Toolbox."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .linalg import logdet_spd, solve_spd


@dataclass(frozen=True)
class FamilyOps:
    """Likelihood primitives for one outcome distribution.

    Every routine here takes the *natural parameter* ``eta`` returned by the
    model's observation function -- a mean for Gaussian outcomes, a logit for
    Bernoulli, a vector of logits for categorical -- and never a probability.
    Keeping the link function here rather than in the model is what lets the
    optimiser work in an unconstrained space.

    ``observation_covariance`` (``R``) is used by the Gaussian family only; the
    discrete families carry their dispersion in the outcome distribution
    itself.
    """

    name: str

    def mean(self, eta, observation_covariance=None):
        """Expected outcome: the mean, ``sigmoid(eta)``, or ``softmax(eta)``.

        This is what a fit reports as its trial-wise prediction, so Bernoulli
        and categorical predictions come back as probabilities even though the
        model works in logits.
        """
        eta = jnp.atleast_1d(eta)
        if self.name == "gaussian":
            return eta
        if self.name == "bernoulli":
            return jax.nn.sigmoid(eta)
        if self.name == "categorical":
            return jax.nn.softmax(eta)
        raise ValueError(f"unsupported family {self.name!r}")

    def log_prob(self, y, eta, observation_covariance=None):
        """Log density (Gaussian) or log mass (Bernoulli, categorical) of ``y``.

        The Bernoulli form ``y*eta - softplus(eta)`` and the categorical form
        ``eta[y] - logsumexp(eta)`` are used instead of taking the log of a
        probability, because both stay finite for extreme logits where the
        probability itself would round to 0 or 1.
        """
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
        """Gradient of ``log_prob`` with respect to ``eta``.

        For all three families this is the prediction error: the residual
        weighted by ``R`` for Gaussian outcomes, and ``observed - expected``
        for the discrete ones. Used by the filter's Fisher-scoring update.
        """
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
        """Fisher information of ``eta``: minus the expected second derivative.

        This is the curvature the filter uses inside a trial update. It is
        deliberately *not* the observed Hessian used for the Laplace posterior
        over static parameters -- the two are never interchanged.
        """
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
        """Draw one outcome from the family, for predictive checks."""
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
    """Look up the likelihood primitives for a family name.

    Raises ``ValueError`` for anything outside ``gaussian``, ``bernoulli`` and
    ``categorical``, so a typo in ``StateModel(family=...)`` fails at
    construction rather than silently fitting something else.
    """
    family = str(name).lower()
    if family not in {"gaussian", "bernoulli", "categorical"}:
        raise ValueError("family must be 'gaussian', 'bernoulli', or 'categorical'")
    return FamilyOps(family)
