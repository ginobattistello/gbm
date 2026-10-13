"""Independent numerical checks for vectorization, scaling and cache reuse."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gbmtoolbox import Config, GammaPrecisionPrior, GaussianPrior, Priors, StateModel, individual_fit
from gbmtoolbox.families import get_family
from gbmtoolbox.optimization import _compiled_objective_functions, _objective_context
from gbmtoolbox.state_model import compiled_evaluator, prepare_subject_data


def reference_scan(model, parameters, data):
    """Original sequential likelihood, retaining dense Gaussian algebra."""
    theta, phi, _, rho = model.unpack_parameters(parameters)
    family = get_family(model.family)

    def step(x, inputs):
        y, u = inputs
        eta = jnp.atleast_1d(model.observation(x, phi, u))
        R = model.observation_covariance_jax(phi, rho, u, eta.size) if model.family == "gaussian" else None
        ll = family.log_prob(y, eta, R)
        return model.evolution(x, theta, u, y), ll

    if data["u"] is None:
        _, ll = jax.lax.scan(lambda x, y: step(x, (y, None)), jnp.asarray(model.initial_state), data["y"])
    else:
        _, ll = jax.lax.scan(step, jnp.asarray(model.initial_state), (data["y"], data["u"]))
    return jnp.sum(ll)


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "categorical"])
@pytest.mark.parametrize("with_inputs", [False, True])
def test_independent_trials_match_original_likelihood_and_derivatives(family, with_inputs):
    def observation(x, phi, u):
        eta = phi[0] + x[0] + (phi[1] * u["stimulus"] if u is not None else phi[1] ** 2)
        return jnp.array([0.0, eta, -0.4 * eta]) if family == "categorical" else eta

    model = StateModel(observation=observation, family=family, initial_state=[0.2],
                       priors=Priors(observation=GaussianPrior([0.1, 0.3], [1.0, 0.4])))
    y = np.linspace(-0.2, 0.8, 9) if family == "gaussian" else np.arange(9) % (3 if family == "categorical" else 2)
    data = prepare_subject_data({"y": y, "u": {"stimulus": np.linspace(-1, 1, 9)} if with_inputs else None})
    p = jnp.asarray(model.parameter_layout.mean)
    fast = lambda q: jnp.sum(model.evaluate_jax(q, data)["loglik"])
    slow = lambda q: reference_scan(model, q, data)
    for operation in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_allclose(operation(fast)(p), operation(slow)(p), rtol=2e-11, atol=2e-11)
    assert "scan[" not in str(jax.make_jaxpr(fast)(p))


@pytest.mark.parametrize("covariance", [0.5, [0.5, 1.2], [[0.5, 0.15], [0.15, 1.2]], "diagonal", "custom"])
def test_multivariate_gaussian_fast_and_fallback_paths(covariance):
    if isinstance(covariance, str) and covariance == "custom":
        covariance = lambda phi, u: jnp.diag(jnp.exp(phi[:2]))
    kwargs = {"process_covariance": None, "observation_covariance": covariance}
    if isinstance(covariance, str):
        kwargs["observation_noise_prior"] = GammaPrecisionPrior([1.0, 2.0], [1.0, 0.5])
    model = StateModel(observation=lambda x, phi, u: phi[:2] + u, observation_dim=2,
                       priors=Priors(observation=GaussianPrior([0.2, -0.1], [1.0, 0.3])), **kwargs)
    data = prepare_subject_data({"y": np.arange(12).reshape(6, 2) / 10, "u": np.linspace(-0.5, 0.5, 6)})
    p = jnp.asarray(model.parameter_layout.mean)
    fast = lambda q: jnp.sum(model.evaluate_jax(q, data)["loglik"])
    slow = lambda q: reference_scan(model, q, data)
    np.testing.assert_allclose(fast(p), slow(p), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.grad(fast)(p), jax.grad(slow)(p), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(jax.hessian(fast)(p), jax.hessian(slow)(p), rtol=1e-11, atol=1e-11)


def regression_model():
    return StateModel(observation=lambda x, phi, u: phi[0] + phi[1] * u,
                      priors=Priors(observation=GaussianPrior([0.1, -0.2], [[4.0, 0.3], [0.3, 0.1]])),
                      observation_covariance=0.4)


@pytest.mark.parametrize("bounds", [None, [(None, 0.25), (-0.5, None)]])
def test_scaled_fit_matches_analytic_gaussian_solution(bounds):
    model = regression_model()
    u = np.linspace(-1, 1, 30)
    y = 0.6 + 0.8*u + 0.1*np.cos(np.arange(30))
    data = [{"y": y, "u": u}]
    fits = [individual_fit(data, model, config=Config(num_init=1, verbose=False, parameter_scaling=scale, hard_bounds=bounds))
            for scale in (False, True)]
    a, b = fits
    np.testing.assert_allclose(a.output.parameters, b.output.parameters, atol=3e-6)
    np.testing.assert_allclose(a.math.hessian[0], b.math.hessian[0], atol=1e-10)
    np.testing.assert_allclose(a.math.covariance[0], b.math.covariance[0], atol=1e-12)
    np.testing.assert_allclose(a.output.log_evidence, b.output.log_evidence, atol=1e-8)
    X = np.column_stack([np.ones(len(u)), u])
    precision = np.linalg.inv(model.priors.observation.covariance)
    H = X.T@X / 0.4 + precision
    np.testing.assert_allclose(b.math.hessian[0], H, atol=1e-10)
    if bounds is None:
        expected = np.linalg.solve(H, X.T@y/0.4 + precision@model.priors.observation.mean)
        np.testing.assert_allclose(b.output.parameters[0], expected, atol=3e-6)
        assert b.math.diagnostics[0].parameter_scaling == "cholesky"
    else:
        assert b.output.parameters[0, 0] == pytest.approx(0.25, abs=1e-10)
        assert b.math.diagnostics[0].parameter_scaling == "diagonal"


def test_cache_reuse_updates_gaussian_and_exact_gamma_priors():
    model = replace(regression_model(), observation_covariance=None, observation_noise_prior=GammaPrecisionPrior(1, 1))
    changed = replace(model, priors=Priors(observation=GaussianPrior([1.0, 0.2], [2.0, 0.3])),
                      observation_noise_prior=GammaPrecisionPrior(3.0, 0.2))
    cfg = Config(num_init=1, verbose=False)
    idx = np.flatnonzero(model.parameter_layout.free_mask)
    assert _compiled_objective_functions(model, idx, cfg) is _compiled_objective_functions(changed, idx, cfg)
    assert compiled_evaluator(model, cfg) is compiled_evaluator(changed, cfg)
    data = [{"y": np.array([0.1, -0.3, 0.5, 0.7, 0.4]), "u": np.linspace(-1, 1, 5)}]
    individual_fit(data, model, config=cfg)
    reused = individual_fit(data, changed, config=cfg)
    # Equivalent new callable forces an independent executable.
    independent = replace(changed, observation=lambda x, phi, u: phi[0] + phi[1]*u)
    fresh = individual_fit(data, independent, config=cfg)
    np.testing.assert_allclose(reused.output.parameters, fresh.output.parameters, atol=1e-10)
    np.testing.assert_allclose(reused.math.hessian[0], fresh.math.hessian[0], atol=1e-9)
    np.testing.assert_allclose(reused.output.log_evidence, fresh.output.log_evidence, atol=1e-10)


def test_all_subjects_prepare_once_and_report_timing(monkeypatch):
    import gbmtoolbox.validation as validation
    calls = []
    original = validation.prepare_subject_data
    def counted(subject):
        calls.append(subject)
        return original(subject)
    monkeypatch.setattr(validation, "prepare_subject_data", counted)
    data = [{"y": np.linspace(0, 1, n), "u": np.linspace(-1, 1, n)} for n in (7, 7, 11)]
    fit = individual_fit(data, regression_model(), config=Config(num_init=1, verbose=False))
    assert len(calls) == len(data)
    assert set(fit.profile.timings) == {"validation", "preflight", "total"}
    assert all(np.isfinite(t) and t >= 0 for t in fit.profile.timings.values())
    for diag in fit.math.diagnostics:
        assert diag.n_evaluations == sum(s.n_evaluations for s in diag.starts)
        assert diag.n_evaluations > 0
        assert all(np.isfinite(t) and t >= 0 for t in diag.timings.values())


def test_uncertain_identity_state_still_requires_filtering():
    model = StateModel(observation=lambda x, phi, u: x[0]+phi[0], initial_state_covariance=0.3,
                       observation_covariance=0.4, priors=Priors(observation=GaussianPrior([0], [1])))
    out = model.evaluate(model.parameter_layout.mean, {"y": np.array([0.2, 0.4]), "u": None})
    assert out["filtering_method"] == "generalized_gaussian_fisher"
    assert np.any(out["state_covariance"] != 0)


def test_fixed_entries_remain_fixed_under_scaling():
    model = StateModel(observation=lambda x, phi, u: phi[0]+phi[1]*u, observation_covariance=0.4,
                       priors=Priors(observation=GaussianPrior([0.2, 0.0], [0.0, 4.0])))
    fit = individual_fit([{"y": np.array([0.1, 0.5, 0.8]), "u": np.array([-1, 0, 1])}], model,
                         config=Config(num_init=1, verbose=False))
    assert fit.output.parameters[0, 0] == 0.2
    assert fit.math.covariance[0][0, 0] == 0


def test_scaling_hessian_equals_direct_original_coordinate_hessian():
    model = replace(regression_model(), observation_covariance=None)
    prepared = prepare_subject_data({"y": np.array([0.2, 0.5, -0.1]), "u": np.array([-1, 0, 1])})
    cfg = Config(num_init=1, verbose=False)
    contexts = [_objective_context(model, replace(cfg, parameter_scaling=scale), prepared) for scale in (False, True)]
    x = np.array([0.3, 0.2, -0.1])
    values, hessians = [], []
    for context in contexts:
        z = context["inverse"]@(x-context["offset"])
        values.append(float(context["compiled"]["value"](jnp.asarray(z), *context["args"])))
        H = np.asarray(context["compiled"]["hessian"](jnp.asarray(z), *context["args"]))
        hessians.append(context["inverse"].T@H@context["inverse"])
    np.testing.assert_allclose(values[0], values[1], atol=1e-12)
    np.testing.assert_allclose(hessians[0], hessians[1], atol=1e-10)
