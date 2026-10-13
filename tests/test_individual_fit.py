import numpy as np
import pytest

from gbmtoolbox import Config, GaussianPrior, HBIConfig, Priors, StateModel, bms, hbi_main, individual_fit


def test_binary_individual_fit_and_propagated_uncertainty(binary_model, binary_data):
    fit = individual_fit(binary_data, binary_model, config=Config(num_init=2, verbose=False, latent_uncertainty="propagated", latent_samples=80))
    assert fit.output.parameters.shape == (1, 2)
    assert np.isfinite(fit.output.log_evidence[0])
    est = fit.output.latent[0]["state"]
    assert est["uncertainty_type"] == "propagated"
    assert est["covariance"].shape == (30, 2, 2)
    assert est["interval_low"].shape == (30, 2)


def test_filtered_latent_output(gaussian_filter_model):
    y = np.array([0.2, -0.1, 0.3, 0.0])
    fit = individual_fit([{"y": y, "u": None}], gaussian_filter_model, config=Config(num_init=2, verbose=False, latent_uncertainty="filtered"))
    est = fit.output.latent[0]["state"]
    assert est["uncertainty_type"] == "filtered"
    assert est["interval_low"] is None
    assert est["covariance"].shape == (4, 1, 1)


def test_all_parameters_fixed_skips_optimizer(binary_data):
    model = StateModel(lambda x, th, u, y: x, lambda x, ph, u: 0.5, "bernoulli", Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])), [0])
    fit = individual_fit(binary_data, model, config=Config(verbose=False))
    assert fit.math.hessian[0].shape == (0, 0)
    assert fit.math.diagnostics[0].laplace_valid


def test_fixed_parameter_has_zero_posterior_variance(binary_data):
    def obs(x, ph, u):
        return ph[0]  # Bernoulli logit

    model = StateModel(lambda x, th, u, y: x, obs, "bernoulli", Priors(GaussianPrior([0], [0]), GaussianPrior([0], [1])), [0])
    fit = individual_fit(binary_data, model, config=Config(num_init=2, verbose=False))
    cov = fit.math.covariance[0]
    assert cov[0, 0] == 0


def test_display_false_fit_remains_plottable(binary_model, binary_data):
    import matplotlib

    matplotlib.use("Agg")
    fit = individual_fit(binary_data, binary_model, config=Config(num_init=1, verbose=False, display=False))
    fig = fit.plot(subject=0, display=False)
    assert len(fig.axes) == 6


def test_unconverged_map_does_not_return_evidence_or_posterior_covariance():
    model = StateModel(observation=lambda x, p, u: p[0], priors=Priors(observation=GaussianPrior([0], [1])), observation_covariance=1.0)
    out = individual_fit([{"y": np.ones(10) * 10, "u": None}], model, config=Config(num_init=1, maxiter=1, verbose=False))
    diag = out.math.diagnostics[0]
    assert not diag.map_converged and not diag.laplace_valid
    assert diag.hess_positive_definite
    assert np.isnan(out.output.log_evidence[0])
    assert out.math.covariance[0] is None
    with pytest.raises(ValueError, match="finite"):
        bms(np.column_stack([out.output.log_evidence, np.zeros(1)]))
    with pytest.raises(RuntimeError, match="valid Laplace"):
        hbi_main(
            [{"y": np.ones(10) * 10, "u": None}],
            [model],
            fit_config=Config(num_init=1, maxiter=1, verbose=False),
            hbi_config=HBIConfig(maxiter=1, verbose=False),
        )


def test_search_path_log_joint_is_recorded_without_display():
    rng = np.random.default_rng(1)
    u = rng.normal(size=30)
    model = StateModel(
        observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t, priors=Priors(observation=GaussianPrior([0.0, 0.0], [10.0, 10.0])), observation_covariance=1.0
    )
    fit = individual_fit([{"y": 1.0 + 2.0 * u, "u": u}], model, config=Config(num_init=2, verbose=False, display=False))
    diag = fit.math.diagnostics[0]
    assert diag.search_log_joint is not None
    assert len(diag.search_log_joint) == len(diag.search_path)
    assert np.all(np.isfinite(diag.search_log_joint))
    assert diag.search_log_joint[-1] == pytest.approx(fit.math.log_joint[0], abs=1e-8)


def test_subject_input_field_is_optional():
    model = StateModel(observation=lambda x, phi, u_t: phi[0], priors=Priors(observation=GaussianPrior([0.0], [1.0])), observation_covariance=1.0)
    y = np.array([0.2, -0.1, 0.4])
    without_u = individual_fit([{"y": y}], model, config=Config(num_init=1, verbose=False))
    with_none = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=1, verbose=False))
    np.testing.assert_array_equal(without_u.output.parameters, with_none.output.parameters)


def _linear_model():
    """Gaussian regression on a scalar trial input."""
    return StateModel(observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t, priors=Priors(observation=GaussianPrior([0.0, 0.0], [10.0, 10.0])))


def test_single_subject_vector_matches_dict_form():
    rng = np.random.default_rng(12)
    u = rng.normal(size=40)
    y = 0.3 + 0.8 * u + rng.normal(scale=0.5, size=40)
    cfg = Config(num_init=2, random_state=5, verbose=False)
    vector = individual_fit(y, _linear_model(), u=u, config=cfg)
    listed = individual_fit([{"y": y, "u": u}], _linear_model(), config=cfg)
    np.testing.assert_array_equal(vector.output.parameters, listed.output.parameters)
    np.testing.assert_array_equal(vector.output.log_evidence, listed.output.log_evidence)
    np.testing.assert_array_equal(vector.data[0]["y"], y)
    np.testing.assert_array_equal(vector.data[0]["u"], u)


def test_single_subject_vector_accepts_multidimensional_inputs():
    rng = np.random.default_rng(13)
    U = rng.normal(size=(60, 2, 3))  # one 2x3 matrix per trial
    w = np.array([0.5, -1.0, 2.0])
    y = U[:, 0, :] @ w + 0.3 * U[:, 1, 0] + rng.normal(scale=0.2, size=60)
    model = StateModel(observation=lambda x, phi, u_t: u_t[0] @ phi[:3] + phi[3] * u_t[1, 0], priors=Priors(observation=GaussianPrior(np.zeros(4), 10.0)))
    fit = individual_fit(y, model, u=U, config=Config(num_init=1, verbose=False))
    np.testing.assert_allclose(fit.output.observation_parameters[0], [0.5, -1.0, 2.0, 0.3], atol=0.1)


def test_single_subject_plain_list_of_outcomes_is_a_vector():
    y = [0.2, -0.1, 0.4, 0.0]
    model = StateModel(observation=lambda x, phi, u_t: phi[0], priors=Priors(observation=GaussianPrior([0.0], [1.0])), observation_covariance=1.0)
    fit = individual_fit(y, model, config=Config(num_init=1, verbose=False))
    assert fit.output.parameters.shape == (1, 1)


def test_ambiguous_vector_inputs_are_rejected():
    model = _linear_model()
    y, u = np.zeros(5), np.zeros(5)
    with pytest.raises(ValueError, match="own dict"):
        individual_fit([{"y": y, "u": u}], model, u=u)
    with pytest.raises(ValueError, match="list of"):
        individual_fit([y, y], model)
    binary = StateModel(observation=lambda x, phi, u_t: phi[0], family="bernoulli", priors=Priors(observation=GaussianPrior([0.0], [1.0])))
    with pytest.raises(ValueError, match="one entry per trial"):
        individual_fit(np.zeros((3, 5)), binary)
