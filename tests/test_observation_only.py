"""Observation-only models and the scale-based observation-noise prior."""

import jax.numpy as jnp
import numpy as np
import pytest

from gbmtoolbox import (
    Config,
    GaussianPrior,
    Priors,
    StateModel,
    individual_fit,
    observation_noise_prior_from_scale,
    robust_scale,
)


def _obs(x, phi, u_t):
    return phi[0] + phi[1] * u_t


def _gaussian_model(**kwargs):
    return StateModel(
        observation=_obs,
        family="gaussian",
        priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])),
        initial_state=[0.0],
        **kwargs,
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

    fit = individual_fit(
        [{"y": y, "u": u}],
        _gaussian_model(),
        config=Config(num_init=3, random_state=0, display=False),
    )
    assert fit.output.evolution_parameters.shape == (1, 0)
    assert fit.output.observation_parameters[0] == pytest.approx([0.5, 1.2], abs=0.2)
    assert fit.output.observation_noise_sd[0, 0] == pytest.approx(0.6, abs=0.15)


def test_evolution_function_with_empty_prior_is_rejected():
    # An evolution prior that declares parameters but no function is a mistake.
    with pytest.raises(ValueError, match="evolution"):
        StateModel(
            observation=_obs,
            family="gaussian",
            priors=Priors(
                evolution=GaussianPrior([0.0], [1.0], names=["gain"]),
                observation=GaussianPrior([0.0], [1.0], names=["b0"]),
            ),
            initial_state=[0.0],
        )


def test_observation_function_is_required():
    with pytest.raises(ValueError, match="observation function"):
        StateModel(
            family="gaussian",
            priors=Priors(observation=GaussianPrior([0.0], [1.0], names=["b0"])),
            initial_state=[0.0],
        )


@pytest.mark.parametrize("family,observation,obs_dim,prior", [
    ("bernoulli", _obs, None, GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["b0", "b1"])),
    (
        "categorical",
        lambda x, phi, u_t: jnp.array([0.0, phi[0] + phi[1] * u_t]),
        2,
        GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["a1", "b1"]),
    ),
])
def test_observation_only_discrete_families(family, observation, obs_dim, prior):
    rng = np.random.default_rng(1)
    T = 80
    u = rng.normal(size=T)
    y = (rng.random(T) < 0.5).astype(int)

    model = StateModel(
        observation=observation,
        family=family,
        priors=Priors(observation=prior),
        initial_state=[0.0],
        observation_dim=obs_dim,
    )
    fit = individual_fit([{"y": y, "u": u}], model, config=Config(num_init=2, random_state=0, display=False))

    # predictions are probabilities, not logits
    pred = np.atleast_2d(fit.output.prediction[0])
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)
    if family == "categorical":
        assert pred.sum(axis=1) == pytest.approx(np.ones(T))


def test_scale_based_noise_prior_tracks_data_scale():
    rng = np.random.default_rng(2)
    for scale in (0.05, 1.0, 50.0):
        prior = observation_noise_prior_from_scale(rng.normal(0.0, scale, 2000))
        # centred near log(scale), up to the Gamma moment-matching offset
        assert prior.mean[0] == pytest.approx(np.log(scale) + 0.2886, abs=0.15)


def test_scale_based_noise_prior_handles_degenerate_outcome():
    # A constant outcome carries no scale information: fall back to the
    # neutral centre rather than log(0) = -inf.
    assert observation_noise_prior_from_scale(np.zeros(10)).mean[0] == 0.0
    assert observation_noise_prior_from_scale(np.array([3.0])).mean[0] == 0.0


def test_prior_sample_size_controls_prior_strength():
    weak = observation_noise_prior_from_scale(prior_scale=1.0, prior_sample_size=2.0)
    strong = observation_noise_prior_from_scale(prior_scale=1.0, prior_sample_size=10.0)
    # more pseudo-observations => tighter prior
    assert strong.covariance[0, 0] < weak.covariance[0, 0]
    # both imply a prior mean precision of 1/s0^2 = 1, so both centre near 0
    assert abs(strong.mean[0]) < abs(weak.mean[0])


def test_prior_scale_and_data_agree():
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 3.0, 4000)
    from_data = observation_noise_prior_from_scale(y)
    from_scale = observation_noise_prior_from_scale(prior_scale=robust_scale(y))
    assert from_data.mean[0] == pytest.approx(from_scale.mean[0], abs=1e-12)


def test_robust_scale_resists_outliers():
    rng = np.random.default_rng(8)
    clean = rng.normal(0.0, 1.0, 400)
    contaminated = clean.copy()
    idx = rng.choice(400, 20, replace=False)  # 5% gross outliers
    contaminated[idx] += rng.normal(0.0, 30.0, 20)

    # the standard deviation is badly inflated; the robust scale is not
    assert np.std(contaminated) > 2.0
    assert robust_scale(contaminated) == pytest.approx(1.0, abs=0.2)


def test_invalid_prior_arguments_are_rejected():
    with pytest.raises(ValueError, match="y or prior_scale"):
        observation_noise_prior_from_scale()
    with pytest.raises(ValueError, match="prior_scale"):
        observation_noise_prior_from_scale(prior_scale=0.0)
    with pytest.raises(ValueError, match="prior_sample_size"):
        observation_noise_prior_from_scale(prior_scale=1.0, prior_sample_size=0.0)


def test_scaled_noise_prior_recovers_sd_on_large_scale_data():
    rng = np.random.default_rng(3)
    true_sd = 40.0
    T = 25
    u = rng.normal(size=T)
    y = rng.normal(0.0, true_sd, T)

    model = StateModel(
        observation=_obs,
        family="gaussian",
        priors=Priors(observation=GaussianPrior([0.0, 0.0], [100.0, 100.0], names=["b0", "b1"])),
        initial_state=[0.0],
        observation_noise_prior=observation_noise_prior_from_scale(y),
    )
    fit = individual_fit([{"y": y, "u": u}], model, config=Config(num_init=3, random_state=0, display=False))
    # within 40% at only 25 trials, without the default prior dragging it to ~1
    assert fit.output.observation_noise_sd[0, 0] == pytest.approx(true_sd, rel=0.4)


def test_default_noise_prior_is_vba_gamma_moment_matched():
    """The default must equal VBA's Ga(1,1) precision prior on the log scale."""
    from gbmtoolbox.parameters import default_observation_noise_prior

    default = default_observation_noise_prior(1)
    vba = observation_noise_prior_from_scale(prior_scale=1.0, prior_sample_size=2.0)

    assert default.mean[0] == pytest.approx(vba.mean[0])
    assert default.covariance[0, 0] == pytest.approx(vba.covariance[0, 0])

    # Moment matching preserves the first two moments, not the quantiles, so
    # the implied 95% SD interval is [0.38, 4.69] rather than the [0.52, 6.29]
    # of the exact Ga(1,1). The two priors are close, not identical.
    from scipy.stats import norm

    lo, hi = np.exp(norm.ppf([0.025, 0.975], default.mean[0], np.sqrt(default.covariance[0, 0])))
    assert lo == pytest.approx(0.38, abs=0.02)
    assert hi == pytest.approx(4.69, abs=0.10)


def test_initial_state_is_optional_for_observation_only_models():
    model = StateModel(
        observation=_obs,
        family="gaussian",
        priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])),
    )
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
        assert other.output.observation_parameters == pytest.approx(
            reference.output.observation_parameters, abs=1e-9
        )
        assert other.output.log_evidence == pytest.approx(reference.output.log_evidence, abs=1e-9)
