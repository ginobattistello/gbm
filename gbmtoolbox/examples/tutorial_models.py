"""Small model/simulation helpers shared by the GBM Toolbox tutorials."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from gbmtoolbox import GaussianPrior, Priors, StateModel


def binary_learning_model(*, fixed_alpha=False):
    """Two-option Rescorla-Wagner model with Bernoulli choices."""

    def evolution(x, theta, u_t, y_t):
        alpha = jax.nn.sigmoid(theta[0])
        choice = y_t.astype(jnp.int32)
        reward = u_t["reward"]
        return x.at[choice].add(alpha * (reward - x[choice]))

    def observation(x, phi, u_t):
        beta = jnp.exp(phi[0])
        return beta * (x[1] - x[0])  # Bernoulli logit

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(GaussianPrior([0.0], [0.0 if fixed_alpha else 1.0], names=["alpha_raw"]), GaussianPrior([1.0], [1.0], names=["log_beta"])),
        initial_state=[0.5, 0.5],
        state_names=["Q0", "Q1"],
        name="two-option learning",
    )


def simulate_binary_subject(rng, n_trials=80, alpha=0.30, beta=3.0, block_length=20):
    q = np.array([0.5, 0.5])
    y = np.zeros(n_trials, dtype=int)
    reward = np.zeros(n_trials)
    for t in range(n_trials):
        reward_probability = np.array([0.75, 0.25]) if (t // block_length) % 2 == 0 else np.array([0.25, 0.75])
        p1 = expit(beta * (q[1] - q[0]))
        choice = int(rng.random() < p1)
        r = float(rng.random() < reward_probability[choice])
        y[t] = choice
        reward[t] = r
        q[choice] += alpha * (r - q[choice])
    return {"y": y, "u": {"reward": reward}}


def continuous_model():
    """Deterministic one-state dynamics with internally estimated Gaussian R."""

    def evolution(x, theta, u_t, y_t):
        return jnp.asarray([theta[0] * x[0] + u_t])

    def observation(x, phi, u_t):
        return x[0] + phi[0]

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(evolution=GaussianPrior([0.60], [0.50], names=["gain"]), observation=GaussianPrior([0.0], [1.0], names=["offset"])),
        initial_state=[0.0],
        # Gaussian outcomes estimate their observation noise by default; pass
        # observation_covariance only to fix R at a known measurement error.
        state_names=["x"],
        name="continuous state model",
    )


def simulate_continuous_subject(rng, n_trials=60, gain=0.70, offset=0.0, sigma=0.50):
    u = rng.normal(size=n_trials)
    x = 0.0
    y = np.zeros(n_trials)
    for t in range(n_trials):
        y[t] = x + offset + rng.normal(0.0, sigma)
        x = gain * x + u[t]
    return {"y": y, "u": u}


def filtered_continuous_model():
    """Linear-Gaussian hidden state with estimated diagonal Q and R."""

    def evolution(x, theta, u_t, y_t):
        return jnp.asarray([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return x[0] + phi[0]

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(GaussianPrior([0.8], [0.2], names=["persistence"]), GaussianPrior([0.0], [1.0], names=["offset"])),
        initial_state=[0.0],
        initial_state_covariance=[1.0],
        process_covariance="diagonal",
        process_noise_prior=GaussianPrior([np.log(np.sqrt(0.10))], [0.5], names=["log_process_sd"]),
        # Process noise is opt-in and needs its own prior; observation noise is
        # estimated by default, so R needs no argument here.
        state_names=["hidden_state"],
        name="stochastic continuous state model",
    )
