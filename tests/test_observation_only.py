"""Observation-only models: structure, defaults and the empty evolution block."""

import jax.numpy as jnp
import numpy as np
import pytest

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def _obs(x, phi, u_t):
    return phi[0] + phi[1] * u_t


def _gaussian_model(**kwargs):
    return StateModel(
        observation=_obs, family="gaussian", priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])), initial_state=[0.0], **kwargs
    )


def test_priors_without_evolution_is_empty_block():
    priors = Priors(observation=GaussianPrior([0.0], [1.0], names=["slope"]))
    assert priors.evolution.dim == 0
    assert priors.dim == 1
    assert priors.names == ("slope",)


def test_empty_prior_has_no_parameters():
    empty = GaussianPrior.empty()
    assert empty.dim == 0
    assert empty.mean.shape == (0,)
    assert empty.covariance.shape == (0, 0)


def test_observation_only_model_needs_no_evolution():
    model = _gaussian_model()
    # no dummy 'unused' parameter is introduced
    assert model.parameter_layout.names == ("b0", "b1", "log_observation_sd")
    assert model.n_theta == 0


def test_observation_only_model_recovers_parameters():
    rng = np.random.default_rng(0)
    T = 200
    u = rng.normal(size=T)
    y = 0.5 + 1.2 * u + rng.normal(0.0, 0.6, T)

    fit = individual_fit([{"y": y, "u": u}], _gaussian_model(), config=Config(num_init=3, random_state=0, display=False))
    assert fit.output.evolution_parameters.shape == (1, 0)
    assert fit.output.observation_parameters[0] == pytest.approx([0.5, 1.2], abs=0.2)
    assert fit.output.observation_noise_sd[0, 0] == pytest.approx(0.6, abs=0.15)


def test_evolution_function_with_empty_prior_is_rejected():
    # An evolution prior that declares parameters but no function is a mistake.
    with pytest.raises(ValueError, match="evolution"):
        StateModel(
            observation=_obs,
            family="gaussian",
            priors=Priors(evolution=GaussianPrior([0.0], [1.0], names=["gain"]), observation=GaussianPrior([0.0], [1.0], names=["b0"])),
            initial_state=[0.0],
        )


def test_observation_function_is_required():
    with pytest.raises(ValueError, match="observation function"):
        StateModel(family="gaussian", priors=Priors(observation=GaussianPrior([0.0], [1.0], names=["b0"])), initial_state=[0.0])


@pytest.mark.parametrize(
    "family,observation,obs_dim,prior",
    [
        ("bernoulli", _obs, None, GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["b0", "b1"])),
        ("categorical", lambda x, phi, u_t: jnp.array([0.0, phi[0] + phi[1] * u_t]), 2, GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["a1", "b1"])),
    ],
)
def test_observation_only_discrete_families(family, observation, obs_dim, prior):
    rng = np.random.default_rng(1)
    T = 80
    u = rng.normal(size=T)
    y = (rng.random(T) < 0.5).astype(int)

    model = StateModel(observation=observation, family=family, priors=Priors(observation=prior), initial_state=[0.0], observation_dim=obs_dim)
    fit = individual_fit([{"y": y, "u": u}], model, config=Config(num_init=2, random_state=0, display=False))

    # predictions are probabilities, not logits
    pred = np.atleast_2d(fit.output.prediction[0])
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)
    if family == "categorical":
        assert pred.sum(axis=1) == pytest.approx(np.ones(T))


def test_initial_state_is_optional_for_observation_only_models():
    model = StateModel(observation=_obs, family="gaussian", priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])))
    assert model.initial_state == pytest.approx([0.0])
    assert model.n_state == 1


def test_initial_state_value_is_inert_when_observation_ignores_it():
    """With no evolution and an observation that ignores x, x0 cannot matter."""
    rng = np.random.default_rng(4)
    T = 80
    u = rng.normal(size=T)
    y = 0.5 + 1.2 * u + rng.normal(0.0, 0.6, T)
    data = [{"y": y, "u": u}]
    config = Config(num_init=2, random_state=0, display=False)

    def fit_with(initial_state):
        model = StateModel(
            observation=_obs,
            family="gaussian",
            priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])),
            initial_state=initial_state,
        )
        return individual_fit(data, model, config=config)

    reference = fit_with([0.0])
    for initial_state in ([999.0], [-5.0, 3.0]):
        other = fit_with(initial_state)
        assert other.output.observation_parameters == pytest.approx(reference.output.observation_parameters, abs=1e-9)
        assert other.output.log_evidence == pytest.approx(reference.output.log_evidence, abs=1e-9)
