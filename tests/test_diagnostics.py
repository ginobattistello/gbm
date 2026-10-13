import numpy as np
import pytest

from gbmtoolbox import (
    Config,
    GaussianPrior,
    Priors,
    StateModel,
    convergence_diagnostics,
    individual_fit,
    noise_parameter_correlations,
    numerical_local_identifiability,
    posterior_hessian_diagnostics,
    prior_preconditioned_information,
)


def test_diagnostics_run(binary_model, binary_data):
    fit = individual_fit(binary_data, binary_model, config=Config(num_init=2, verbose=False))
    c = convergence_diagnostics(fit)
    assert c.n_starts == 2
    h = posterior_hessian_diagnostics(fit)
    assert h.eigenvalues.size == 2
    info = prior_preconditioned_information(fit)
    assert info.eigenvalues.size == 2
    ident = numerical_local_identifiability(fit)
    assert ident.rank <= 2


def test_local_identifiability_detects_redundant_parameters():
    # logit = phi0 + phi1: only the sum is observable, so the information
    # matrix is rank 1 in a two-parameter model.
    model = StateModel(
        lambda x, th, u, y: x,
        lambda x, ph, u: ph[0] + ph[1],
        "bernoulli",
        Priors(GaussianPrior([0], [0]), GaussianPrior([0, 0], [1, 1], names=["a", "b"])),
        [0],
    )
    data = [{"y": np.array([0, 1, 1, 0, 1, 0]), "u": None}]
    fit = individual_fit(data, model, config=Config(num_init=2, verbose=False))
    ident = numerical_local_identifiability(fit)
    assert ident.rank == 1
    assert ident.nullity == 1


def test_identifiability_returns_the_missing_null_direction():
    model = StateModel(observation=lambda x, p, u: p[0] + p[1], priors=Priors(observation=GaussianPrior([0, 0], [1, 1])), observation_covariance=1.0)
    fit = individual_fit([{"y": np.array([1.0]), "u": None}], model, config=Config(num_init=1, verbose=False))
    out = numerical_local_identifiability(fit)
    assert out.rank == 1 and out.nullity == 1
    assert out.parameter_directions.shape == (2, 2)
    np.testing.assert_allclose(np.array([[1.0, 1.0]]) @ out.parameter_directions[out.rank :].T, 0, atol=1e-12)


def _regression_fit():
    """Two-coefficient Gaussian regression with estimated observation noise."""
    rng = np.random.default_rng(4)
    u = rng.normal(size=40)
    y = 0.5 + 1.2 * u + rng.normal(size=40)
    model = StateModel(
        observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t, priors=Priors(observation=GaussianPrior([0.0, 1.0], [10.0, 0.0], names=["a", "b"]))
    )
    return individual_fit([{"y": y, "u": u}], model, config=Config(num_init=1, verbose=False))


def test_noise_correlations_have_noise_rows_and_free_parameter_columns():
    fit = _regression_fit()
    out = noise_parameter_correlations(fit)
    # "b" is fixed by a zero prior variance, so it has no posterior correlation.
    assert out.noise_names == ("log_observation_sd",)
    assert out.parameter_names == ("a", "log_observation_sd")
    assert out.correlation.shape == (1, 2)
    assert out.correlation[0, 1] == pytest.approx(1.0)
    cov = fit.math.covariance[0]
    expected = cov[2, 0] / np.sqrt(cov[2, 2] * cov[0, 0])
    assert out.correlation[0, 0] == pytest.approx(expected)


def test_convergence_summary_reports_no_starts_when_everything_is_fixed():
    model = StateModel(observation=lambda x, phi, u_t: phi[0] + 0.0 * u_t, priors=Priors(observation=GaussianPrior([0.0], [0.0])), observation_covariance=1.0)
    fit = individual_fit([{"y": np.zeros(5), "u": np.zeros(5)}], model, config=Config(verbose=False))
    summary = convergence_diagnostics(fit)
    assert summary.n_starts == summary.n_successful == summary.n_objective_agreeing == 0
    assert summary.same_solution
