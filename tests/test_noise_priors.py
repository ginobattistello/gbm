"""Observation-noise priors: the default, the scale-based form and robust scale."""

import numpy as np
import pytest

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit, observation_noise_prior_from_scale, robust_scale


def _obs(x, phi, u_t):
    return phi[0] + phi[1] * u_t


def test_scale_based_noise_prior_tracks_data_scale():
    rng = np.random.default_rng(2)
    for scale in (0.05, 1.0, 50.0):
        prior = observation_noise_prior_from_scale(rng.normal(0.0, scale, 2000))
        # log-SD prior mean is log(scale) + (log(nu0/2) - digamma(nu0/2)) / 2
        assert prior.mean[0] == pytest.approx(np.log(scale) + 0.2886, abs=0.15)


def test_scale_based_noise_prior_rejects_degenerate_outcome():
    # A constant outcome carries no scale information: ask for prior_scale
    # rather than inventing a centre.
    for y in (np.zeros(10), np.array([3.0])):
        with pytest.raises(ValueError, match="prior_scale"):
            observation_noise_prior_from_scale(y)


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


def test_default_noise_prior_is_exact_vba_gamma():
    """The scale-based helper reproduces the exact default at s0=1, nu0=2."""
    from gbmtoolbox import GammaPrecisionPrior
    from gbmtoolbox.parameters import default_observation_noise_prior

    default = default_observation_noise_prior(1)
    vba = observation_noise_prior_from_scale(prior_scale=1.0, prior_sample_size=2.0)
    assert isinstance(default, GammaPrecisionPrior)
    assert isinstance(vba, GammaPrecisionPrior)
    np.testing.assert_array_equal(vba.shape, default.shape)
    np.testing.assert_array_equal(vba.rate, default.rate)
    assert vba.names == default.names

    from scipy.stats import gamma

    lo, hi = 1.0 / np.sqrt(gamma.ppf([0.975, 0.025], a=1.0))
    assert lo == pytest.approx(0.52, abs=0.02)
    assert hi == pytest.approx(6.29, abs=0.10)


def test_scale_based_noise_prior_is_conjugate_gamma():
    prior = observation_noise_prior_from_scale(prior_scale=3.0, prior_sample_size=10.0, dim=2)
    np.testing.assert_allclose(prior.shape, [5.0, 5.0])
    np.testing.assert_allclose(prior.rate, [45.0, 45.0])
    assert prior.names == ("log_observation_sd[0]", "log_observation_sd[1]")
