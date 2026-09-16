"""Prior and posterior predictive simulation using the fitted StateModel."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .state_model import trial_input


@dataclass
class PredictiveResult:
    parameters: np.ndarray
    replicated_data: list[dict]


def _draw_gaussian(mean, cov, rng):
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    if mean.size == 0 or np.allclose(cov, 0.0):
        return mean.copy()
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.maximum(vals, 0.0)
    return mean + vecs @ (np.sqrt(vals) * rng.normal(size=mean.size))


def _jaxify_u(u_t):
    if u_t is None:
        return None
    return jax.tree_util.tree_map(jnp.asarray, u_t)


def simulate_subject(model, parameters, template_data, *, rng):
    """Generate one replicated subject from the model's generative equations."""
    theta, phi, rho_q, rho_r = model.unpack_parameters(jnp.asarray(parameters, dtype=jnp.float64))
    y_template = np.asarray(template_data["y"])
    T = y_template.shape[0]
    u = copy.deepcopy(template_data.get("u", None))
    x = _draw_gaussian(model.initial_state, model.initial_covariance_matrix(), rng)
    ys = []
    states = np.zeros((T, model.n_state))

    for t in range(T):
        u_t_host = trial_input(u, t, T)
        u_t = _jaxify_u(u_t_host)
        xj = jnp.asarray(x, dtype=jnp.float64)
        states[t] = x
        eta = jnp.atleast_1d(model.observation(xj, phi, u_t))

        if model.family == "bernoulli":
            p = float(jax.nn.sigmoid(eta[0]))
            y_t = int(rng.random() < p)
        elif model.family == "categorical":
            p = np.asarray(jax.nn.softmax(eta), dtype=float)
            y_t = int(rng.choice(len(p), p=p))
        else:
            mu = np.asarray(eta, dtype=float)
            R = np.asarray(model.observation_covariance_jax(phi, rho_r, u_t, mu.size), dtype=float)
            draw = rng.multivariate_normal(mu, R)
            y_t = float(draw[0]) if mu.size == 1 else draw

        ys.append(y_t)
        x_det = np.asarray(model.evolution(xj, theta, u_t, jnp.asarray(y_t)), dtype=float).reshape(-1)
        Q = np.asarray(model.process_covariance_jax(theta, rho_q, u_t), dtype=float)
        x = _draw_gaussian(x_det, Q, rng)

    return {"y": np.asarray(ys), "u": u, "latent": states}


def prior_predictive(model, template_data, *, n_samples=100, random_state=42) -> PredictiveResult:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(random_state)
    layout = model.parameter_layout
    params = np.stack([_draw_gaussian(layout.mean, layout.covariance, rng) for _ in range(n_samples)])
    reps = [simulate_subject(model, p, template_data, rng=rng) for p in params]
    return PredictiveResult(parameters=params, replicated_data=reps)


def posterior_predictive(fit, subject=0, *, n_samples=100, random_state=42) -> PredictiveResult:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    diag = fit.math.diagnostics[subject]
    cov = fit.math.covariance[subject]
    if not diag.laplace_valid or cov is None:
        raise ValueError("posterior predictive checks require a valid Laplace posterior")
    rng = np.random.default_rng(random_state)
    mean = fit.output.parameters[subject]
    params = np.stack([_draw_gaussian(mean, cov, rng) for _ in range(n_samples)])
    reps = [simulate_subject(fit.model, p, fit.data[subject], rng=rng) for p in params]
    return PredictiveResult(parameters=params, replicated_data=reps)
