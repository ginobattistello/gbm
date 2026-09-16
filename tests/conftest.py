import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gbmtoolbox import GaussianPrior, Priors, StateModel


@pytest.fixture
def binary_model():
    def evolution(x, theta, u_t, y_t):
        alpha = jax.nn.sigmoid(theta[0])
        c = y_t.astype(jnp.int32)
        return x.at[c].add(alpha * (u_t["reward"] - x[c]))

    def observation(x, phi, u_t):
        beta = jnp.exp(phi[0])
        return beta * (x[1] - x[0])  # Bernoulli logit

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(
            GaussianPrior([0.0], [1.0], names=["alpha_raw"]),
            GaussianPrior([1.0], [1.0], names=["log_beta"]),
        ),
        initial_state=[0.5, 0.5],
        state_names=["Q0", "Q1"],
        name="binary_rw",
    )


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    return [{
        "y": rng.integers(0, 2, 30),
        "u": {"reward": rng.integers(0, 2, 30).astype(float)},
    }]


@pytest.fixture
def gaussian_filter_model():
    def evolution(x, theta, u_t, y_t):
        return jnp.asarray([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return jnp.asarray([x[0] + phi[0]])

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(
            GaussianPrior([0.8], [0.2], names=["a"]),
            GaussianPrior([0.0], [1.0], names=["offset"]),
        ),
        initial_state=[0.0],
        initial_state_covariance=[1.0],
        process_covariance=[0.1],
        observation_covariance=[0.25],
        state_names=["x"],
        name="linear_gaussian",
    )
