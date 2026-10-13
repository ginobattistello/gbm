"""Independent analytic checks for the exact Gamma precision prior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import gamma

from gbmtoolbox import (
    Config,
    GammaPrecisionPrior,
    GaussianPrior,
    HBIConfig,
    Priors,
    StateModel,
    bms,
    hbi_main,
    individual_fit,
    prior_preconditioned_information,
    prior_predictive,
)
from gbmtoolbox.parameters import default_observation_noise_prior
from gbmtoolbox.priors import broadcast_prior


def fixed_mean_model(mu=0.0, prior=None):
    return StateModel(observation=lambda x, phi, u: mu, priors=Priors(observation=GaussianPrior([0.0], [0.0])), observation_noise_prior=prior)


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (0.5, 2.0), (3.0, 0.2)])
def test_transformed_density_normalization_and_derivatives(a, b):
    prior = GammaPrecisionPrior(a, b)
    rho = jnp.array([0.3])
    lam = np.exp(-2.0 * float(rho[0]))
    reference = gamma.logpdf(lam, a, scale=1 / b) + np.log(2 * lam)
    assert float(prior.log_prob_jax(rho)) == pytest.approx(reference, abs=1e-12)
    np.testing.assert_allclose(jax.grad(prior.log_prob_jax)(rho), [-2 * a + 2 * b * lam], atol=1e-12)
    np.testing.assert_allclose(jax.hessian(prior.log_prob_jax)(rho), [[-4 * b * lam]], atol=1e-12)
    integral, _ = quad(lambda r: np.exp(float(prior.log_prob_jax(jnp.array([r])))), -20, 50, epsabs=1e-10)
    assert integral == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n,a,b,zero_residual", [(1, 1, 1, False), (8, 0.5, 2, False), (50, 3, 0.2, False), (8, 1, 1, True)])
def test_fit_hessian_evidence_and_information(n, a, b, zero_residual):
    y = np.zeros(n) if zero_residual else np.random.default_rng(n).normal(size=n)
    S = float(y @ y)
    A, B = a + n / 2, b + S / 2
    fit = individual_fit([{"y": y, "u": None}], fixed_mean_model(prior=GammaPrecisionPrior(a, b)), config=Config(num_init=2, verbose=False))
    rho = fit.output.observation_noise_parameters[0, 0]
    assert rho == pytest.approx(0.5 * np.log(B / A), abs=2e-6)
    assert fit.math.hessian[0][0, 0] == pytest.approx(4 * A, rel=5e-6)
    C = -n / 2 * np.log(2 * np.pi) + a * np.log(b) - gammaln(a)
    exact = C + gammaln(A) - A * np.log(B)
    stirling_error = (A - 0.5) * np.log(A) - A + 0.5 * np.log(2 * np.pi) - gammaln(A)
    assert fit.output.log_evidence[0] == pytest.approx(exact + stirling_error, abs=2e-6)
    assert fit.input.observation_noise_prior["distribution"] == "gamma_precision"
    # Whitening uses global log-SD variance, while subtraction uses local curvature.
    variance = fit.input.prior_covariance[-1, -1]
    expected_info = variance * 2 * S * np.exp(-2 * rho)
    info = prior_preconditioned_information(fit)
    assert info.eigenvalues[0] == pytest.approx(expected_info, abs=1e-8)


def test_exact_gamma_sampling_and_broadcasting():
    prior = broadcast_prior(GammaPrecisionPrior([1], [2], names=["noise"]), 2, prefix="log_observation_sd")
    assert isinstance(prior, GammaPrecisionPrior)
    assert prior.names == ("noise[0]", "noise[1]")
    draw = prior.sample(np.random.default_rng(2), 80000)
    precision = np.exp(-2 * draw)
    np.testing.assert_allclose(np.mean(precision, axis=0), [0.5, 0.5], atol=0.008)
    np.testing.assert_allclose(np.quantile(precision, 0.95, axis=0), gamma.ppf(0.95, a=1, scale=0.5), atol=0.03)
    # Public prior-predictive draws must use the exact default, too.
    result = prior_predictive(fixed_mean_model(), {"y": np.zeros(1), "u": None}, n_samples=2000)
    lam = np.exp(-2 * result.parameters[:, -1])
    assert np.mean(lam) == pytest.approx(1, abs=0.08)
    assert np.quantile(lam, 0.95) == pytest.approx(gamma.ppf(0.95, 1), abs=0.2)
    assert np.all(result.parameters[:, 0] == 0)


def test_multivariate_default_remains_exact_and_independent():
    model = StateModel(observation=lambda x, phi, u: jnp.zeros(2), observation_dim=2, priors=Priors(observation=GaussianPrior.empty()))
    y = np.random.default_rng(3).normal(size=(12, 2))
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=1, verbose=False))
    expected = np.sqrt((1 + np.sum(y**2, axis=0) / 2) / (1 + len(y) / 2))
    np.testing.assert_allclose(fit.output.observation_noise_sd[0], expected, rtol=2e-6)
    assert fit.math.hessian[0].shape == (2, 2)
    np.testing.assert_allclose(fit.math.hessian[0], np.eye(2) * (4 + 2 * len(y)), atol=1e-4)


def test_bayes_factor_and_bms_from_analytic_noise_integrals():
    y = np.random.default_rng(15).normal(0.7, 0.4, size=20)
    data = [{"y": y, "u": None}]
    fits = [individual_fit(data, fixed_mean_model(mu), config=Config(num_init=1, verbose=False)) for mu in (0.0, 0.7)]
    A = 1 + len(y) / 2
    exact_bf = -A * np.log((1 + np.sum(y**2) / 2) / (1 + np.sum((y - 0.7) ** 2) / 2))
    lme = np.array([[f.output.log_evidence[0] for f in fits]])
    assert lme[0, 0] - lme[0, 1] == pytest.approx(exact_bf, abs=2e-6)
    original = bms(lme)
    shifted = bms(lme + 123.0)
    np.testing.assert_allclose(original.responsibilities, shifted.responsibilities, atol=1e-12)


@pytest.mark.parametrize("a,b", [(0, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_gamma_hyperparameters(a, b):
    with pytest.raises(ValueError, match="finite and positive"):
        GammaPrecisionPrior(a, b)


def test_default_and_explicit_gamma_are_identical():
    y = np.array([-0.1, 0.4, 0.6])
    data = [{"y": y, "u": None}]
    cfg = Config(num_init=1, verbose=False)
    default = individual_fit(data, fixed_mean_model(), config=cfg)
    explicit = individual_fit(data, fixed_mean_model(prior=default_observation_noise_prior(1)), config=cfg)
    np.testing.assert_allclose(default.output.parameters, explicit.output.parameters, atol=1e-12)
    np.testing.assert_allclose(default.output.log_evidence, explicit.output.log_evidence, atol=1e-12)


def test_hbi_preserves_exact_noise_prior_during_scientific_prior_updates():
    prior = GammaPrecisionPrior(shape=2, rate=0.5)
    model = StateModel(observation=lambda x, phi, u: phi[0], priors=Priors(observation=GaussianPrior([0.0], [1.0])), observation_noise_prior=prior)
    data = [{"y": np.array([0.1, 0.4, 0.2, -0.2]), "u": None}]
    result = hbi_main(data, [model, model], fit_config=Config(num_init=1, verbose=False), hbi_config=HBIConfig(maxiter=2, verbose=False))
    for fit in result.fits:
        resolved = fit.model.parameter_layout.observation_noise_prior
        assert isinstance(resolved, GammaPrecisionPrior)
        np.testing.assert_array_equal(resolved.shape, [2])
        np.testing.assert_array_equal(resolved.rate, [0.5])
        assert np.isfinite(fit.output.log_evidence[0])


def test_consistent_unit_change_preserves_bayes_factor():
    y = np.array([-0.1, 0.4, 0.9, 0.5])
    values = []
    for c in (1.0, 100.0):
        models = [fixed_mean_model(c * mu, GammaPrecisionPrior(1, c**2)) for mu in (0.0, 0.5)]
        values.append(np.array([individual_fit([{"y": c * y, "u": None}], m, config=Config(num_init=1, verbose=False)).output.log_evidence[0] for m in models]))
    np.testing.assert_allclose(values[1], values[0] - len(y) * np.log(100), atol=2e-6)


def test_joint_scientific_parameter_and_noise_evidence_against_quadrature():
    y = np.random.default_rng(5).normal(0.3, 0.8, 40)
    model = StateModel(observation=lambda x, phi, u: phi[0], priors=Priors(observation=GaussianPrior([0.0], [1.0])))
    data = [{"y": y, "u": None}]
    fit = individual_fit(data, model, config=Config(num_init=2, verbose=False))
    fd = individual_fit(data, model, config=Config(num_init=2, verbose=False, hessian_method="central_fd"))
    np.testing.assert_allclose(fit.math.hessian[0], fd.math.hessian[0], rtol=2e-5, atol=2e-5)
    A = 1 + len(y) / 2
    C = -len(y) / 2 * np.log(2 * np.pi) + gammaln(A)

    def log_integrand(mu):
        return C - A * np.log(1 + np.sum((y - mu) ** 2) / 2) - 0.5 * (np.log(2 * np.pi) + mu**2)

    offset = log_integrand(fit.output.observation_parameters[0, 0])
    integral, _ = quad(lambda mu: np.exp(log_integrand(mu) - offset), -10, 10, epsabs=1e-11)
    exact = offset + np.log(integral)
    assert abs(fit.output.log_evidence[0] - exact) < 0.1


def test_filtered_gaussian_default_prior_against_joint_gaussian_likelihood():
    from scipy.stats import multivariate_normal

    y = np.array([0.2, -0.4, 0.7])
    model = StateModel(observation=lambda x, phi, u: x[0], initial_state_covariance=0.5, priors=Priors(observation=GaussianPrior.empty()))
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=2, verbose=False, filter_jitter=0.0))
    rho = fit.output.observation_noise_parameters[0, 0]
    cov = 0.5 * np.ones((len(y), len(y))) + np.exp(2 * rho) * np.eye(len(y))
    expected_ll = multivariate_normal.logpdf(y, mean=np.zeros(len(y)), cov=cov)
    expected_lp = np.log(2) - 2 * rho - np.exp(-2 * rho)
    assert fit.math.log_likelihood[0] == pytest.approx(expected_ll, abs=1e-10)
    assert fit.math.log_prior[0] == pytest.approx(expected_lp, abs=1e-10)
    assert fit.math.diagnostics[0].laplace_valid
