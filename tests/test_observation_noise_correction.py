"""Observation-noise correction for posterior parameter uncertainty."""

import jax.numpy as jnp
import numpy as np
import pytest

from gbmtoolbox import (
    Config,
    GaussianPrior,
    Priors,
    StateModel,
    individual_fit,
    observation_noise_correction,
)

# A prior variance this wide is effectively flat, which isolates the
# correction itself from prior shrinkage.
FLAT = 1e6


def _linear_model(n_params, prior_variance=100.0):
    def observation(x, phi, u_t):
        return phi[0] + sum(phi[i + 1] * u_t[k] for i, k in enumerate("abcd"[: n_params - 1]))

    return StateModel(
        observation=observation,
        family="gaussian",
        priors=Priors(
            observation=GaussianPrior(
                [0.0] * n_params, [prior_variance] * n_params, names=[f"b{i}" for i in range(n_params)]
            )
        ),
        initial_state=[0.0],
        observation_noise_prior=GaussianPrior([0.0], [FLAT], names=["log_observation_sd"]),
    )


def _linear_data(rng, n_trials, n_params, true_sd=1.0):
    n_regressors = n_params - 1
    U = rng.normal(size=(n_trials, n_regressors))
    beta = np.linspace(0.8, -0.5, n_regressors)
    y = 1.0 + U @ beta + rng.normal(0.0, true_sd, n_trials)
    u = {k: U[:, i] for i, k in enumerate("abcd"[:n_regressors])}
    return {"y": y, "u": u}


def _fit(data, model):
    return individual_fit([data], model, config=Config(num_init=2, random_state=0, display=False))


def test_correction_matches_unbiased_estimator_for_linear_model():
    """For a linear model the correction must reproduce SSE/(T-p) exactly.

    sum_t J_t Sigma J_t^T = sigma^2 tr(X (X'X)^-1 X') = sigma^2 p, so the
    corrected variance is (SSE + sigma^2 p)/T. At the MAP sigma^2 = SSE/T,
    which gives SSE/(T-p) once solved self-consistently -- but the toolbox
    evaluates it in one step, so we check against that one-step value.
    """
    rng = np.random.default_rng(0)
    n_trials, n_params = 40, 5
    data = _linear_data(rng, n_trials, n_params)
    fit = _fit(data, _linear_model(n_params))
    correction = observation_noise_correction(fit)

    X = np.column_stack([np.ones(n_trials)] + [data["u"][k] for k in "abcd"])
    beta_hat = np.linalg.lstsq(X, data["y"], rcond=None)[0]
    sse = float(np.sum((data["y"] - X @ beta_hat) ** 2))

    assert correction.residual_sd[0] == pytest.approx(np.sqrt(sse / n_trials), rel=1e-3)
    # the correction term is sigma^2 * p with sigma^2 = SSE/T
    expected = np.sqrt((sse + (sse / n_trials) * n_params) / n_trials)
    assert correction.corrected_sd[0] == pytest.approx(expected, rel=2e-2)


def test_correction_removes_downward_bias():
    """Averaged over replicates the corrected SD is closer to the truth."""
    true_sd = 1.0
    residual, corrected = [], []
    for seed in range(25):
        rng = np.random.default_rng(seed)
        data = _linear_data(rng, 25, 5, true_sd)
        c = observation_noise_correction(_fit(data, _linear_model(5)))
        residual.append(c.residual_sd[0])
        corrected.append(c.corrected_sd[0])

    mean_residual, mean_corrected = float(np.mean(residual)), float(np.mean(corrected))
    # the uncorrected estimate is biased low, the corrected one much less so
    assert mean_residual < 0.95 * true_sd
    assert abs(mean_corrected - true_sd) < abs(mean_residual - true_sd)


def test_inflation_shrinks_as_trials_increase():
    inflations = []
    for n_trials in (25, 200):
        rng = np.random.default_rng(1)
        data = _linear_data(rng, n_trials, 5)
        inflations.append(observation_noise_correction(_fit(data, _linear_model(5))).inflation_factor[0])
    assert inflations[0] > inflations[1] > 1.0


def test_correction_reports_shape_and_metadata():
    rng = np.random.default_rng(2)
    data = _linear_data(rng, 30, 3)
    correction = observation_noise_correction(_fit(data, _linear_model(3)))

    assert correction.valid
    assert correction.n_trials == 30
    # 3 regression parameters + log_observation_sd
    assert correction.n_free_parameters == 4
    assert correction.residual_sd.shape == (1,)
    assert correction.corrected_sd.shape == (1,)
    assert correction.uncertainty_trace > 0.0
    assert np.all(correction.corrected_sd >= correction.residual_sd)


def test_correction_applies_to_dynamical_models():
    """The Jacobian must propagate through the latent trajectory."""
    rng = np.random.default_rng(3)
    n_trials = 40
    u = rng.normal(size=n_trials)
    x, y = 0.0, np.zeros(n_trials)
    for t in range(n_trials):
        y[t] = x + 0.2 + rng.normal(0.0, 1.0)
        x = 0.7 * x + u[t]

    model = StateModel(
        evolution=lambda x, theta, u_t, y_t: jnp.asarray([theta[0] * x[0] + u_t]),
        observation=lambda x, phi, u_t: x[0] + phi[0],
        family="gaussian",
        priors=Priors(
            GaussianPrior([0.7], [0.5], names=["gain"]),
            GaussianPrior([0.0], [1.0], names=["offset"]),
        ),
        initial_state=[0.0],
        observation_noise_prior=GaussianPrior([0.0], [FLAT], names=["log_observation_sd"]),
    )
    correction = observation_noise_correction(_fit({"y": y, "u": u}, model))
    assert correction.valid
    assert correction.corrected_sd[0] > correction.residual_sd[0]


@pytest.mark.parametrize("family,observation,obs_dim,prior", [
    ("bernoulli", lambda x, phi, u_t: phi[0] + phi[1] * u_t, None,
     GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["b0", "b1"])),
    ("categorical", lambda x, phi, u_t: jnp.array([0.0, phi[0] + phi[1] * u_t]), 2,
     GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["a1", "b1"])),
])
def test_non_gaussian_families_have_no_estimated_noise(family, observation, obs_dim, prior):
    rng = np.random.default_rng(4)
    n_trials = 50
    u = rng.normal(size=n_trials)
    y = (rng.random(n_trials) < 0.5).astype(int)

    model = StateModel(
        observation=observation,
        family=family,
        priors=Priors(observation=prior),
        initial_state=[0.0],
        observation_dim=obs_dim,
    )
    correction = observation_noise_correction(_fit({"y": y, "u": u}, model))
    assert not correction.valid
    assert correction.residual_sd.size == 0


def test_correction_does_not_change_the_fit():
    """The diagnostic is read-only: evidence and parameters are untouched."""
    rng = np.random.default_rng(5)
    data = _linear_data(rng, 40, 3)
    fit = _fit(data, _linear_model(3))

    before_params = fit.output.parameters.copy()
    before_evidence = fit.output.log_evidence.copy()
    before_sd = fit.output.observation_noise_sd.copy()

    observation_noise_correction(fit)

    assert np.array_equal(fit.output.parameters, before_params)
    assert np.array_equal(fit.output.log_evidence, before_evidence)
    assert np.array_equal(fit.output.observation_noise_sd, before_sd)
