import jax.numpy as jnp
import numpy as np
import pytest

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel
from gbmtoolbox.validation import validate_fit_spec


def test_rejects_nan_y(binary_model, binary_data):
    bad = [{"y": binary_data[0]["y"].astype(float).copy(), "u": binary_data[0]["u"]}]
    bad[0]["y"][2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_fit_spec(bad, binary_model, Config())


def test_rejects_binary_codes(binary_model, binary_data):
    bad = [{"y": binary_data[0]["y"].copy(), "u": binary_data[0]["u"]}]
    bad[0]["y"][0] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_fit_spec(bad, binary_model, Config())


def test_rejects_mismatched_input_length(binary_model, binary_data):
    bad = [{"y": binary_data[0]["y"], "u": {"reward": np.zeros(5)}}]
    with pytest.raises(ValueError, match="expected 30"):
        validate_fit_spec(bad, binary_model, Config())


def test_filtered_uncertainty_rejected_for_deterministic_model(binary_model, binary_data):
    with pytest.raises(ValueError, match="requires initial-state uncertainty"):
        validate_fit_spec(binary_data, binary_model, Config(latent_uncertainty="filtered"))


def test_filtered_rejected_for_deterministic_jax_model():
    """Regression: a deterministic model yields an identically zero state
    covariance, which would otherwise be reported as genuine filtered
    uncertainty (sd=0 at every trial, labelled uncertainty_type='filtered')."""

    def evolution(x, theta, u_t, y_t):
        alpha = 1.0 / (1.0 + jnp.exp(-theta[0]))
        choice = y_t.astype(jnp.int32)
        return x.at[choice].add(alpha * (u_t["reward"] - x[choice]))

    def observation(x, phi, u_t):
        return jnp.exp(phi[0]) * (x[1] - x[0])

    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(GaussianPrior([0.0], [1.0], names=["alpha_raw"]), GaussianPrior([1.0], [1.0], names=["log_beta"])),
        initial_state=[0.5, 0.5],
        state_names=["Q0", "Q1"],
    )
    assert not model.has_state_uncertainty()
    rng = np.random.default_rng(0)
    data = [{"y": rng.integers(0, 2, 20), "u": {"reward": rng.integers(0, 2, 20).astype(float)}}]
    with pytest.raises(ValueError, match="requires initial-state uncertainty"):
        validate_fit_spec(data, model, Config(latent_uncertainty="filtered"))


def test_rejects_nondeterministic_evolution():
    """A host-side RNG inside evolution is traced once and silently frozen into
    the compiled graph, so it must be rejected before tracing."""

    def evolution(x, theta, u_t, y_t):
        alpha = 1.0 / (1.0 + jnp.exp(-theta[0]))
        choice = y_t.astype(jnp.int32)
        return x.at[choice].add(alpha * (u_t["reward"] - x[choice]) + np.random.randn())

    def observation(x, phi, u_t):
        return jnp.exp(phi[0]) * (x[1] - x[0])

    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(GaussianPrior([0.0], [1.0], names=["alpha_raw"]), GaussianPrior([1.0], [1.0], names=["log_beta"])),
        initial_state=[0.5, 0.5],
        state_names=["Q0", "Q1"],
    )
    rng = np.random.default_rng(0)
    data = [{"y": rng.integers(0, 2, 20), "u": {"reward": rng.integers(0, 2, 20).astype(float)}}]
    with pytest.raises(ValueError, match="evolution must be deterministic"):
        validate_fit_spec(data, model, Config())


def test_latent_sampling_settings_discarded_when_not_propagated():
    cfg = Config(latent_uncertainty="none", latent_samples=55, latent_interval=0.8)
    assert cfg.latent_samples is None
    assert cfg.latent_interval is None
