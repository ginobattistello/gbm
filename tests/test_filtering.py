import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from gbmtoolbox import GaussianPrior, Priors, StateModel
from gbmtoolbox.state_model import prepare_subject_data


def test_linear_gaussian_filter_matches_kalman(gaussian_filter_model):
    model = gaussian_filter_model
    params = np.array([0.8, 0.1])
    y = np.array([0.2, -0.1, 0.3])
    out = model.evaluate(params, {"y": y, "u": None})

    m, P = 0.0, 1.0
    means, covs, lls = [], [], []
    a, offset, Q, R = 0.8, 0.1, 0.1, 0.25
    for yt in y:
        S = P + R
        lls.append(-0.5 * (np.log(2 * np.pi * S) + (yt - (m + offset)) ** 2 / S))
        K = P / S
        m = m + K * (yt - (m + offset))
        P = (1 - K) * P
        means.append(m)
        covs.append(P)
        m = a * m
        P = a * a * P + Q
    np.testing.assert_allclose(out["states"][:, 0], means, atol=1e-7)
    np.testing.assert_allclose(out["state_covariance"][:, 0, 0], covs, atol=1e-7)
    np.testing.assert_allclose(out["loglik"], lls, atol=1e-7)


def test_bernoulli_laplace_predictive_close_to_quadrature():
    # Single uncertain state, no evolution needed for the one trial.
    # The observation returns the logit; families.py applies the sigmoid, so
    # the implied P(y=1) is expit(x[0]) exactly as in the quadrature reference.
    model = StateModel(
        evolution=lambda x, th, u, y: x,
        observation=lambda x, ph, u: x[0],
        family="bernoulli",
        priors=Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])),
        initial_state=[0.4],
        initial_state_covariance=[0.5],
        process_covariance=None,
    )
    out = model.evaluate([0, 0], {"y": np.array([1]), "u": None})
    sd = np.sqrt(0.5)
    mu = 0.4
    f = lambda x: expit(x) * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    exact = quad(f, -np.inf, np.inf, epsabs=1e-12)[0]
    assert abs(np.exp(out["loglik"][0]) - exact) < 0.04
    # Renamed in the JAX rewrite; one unified filter now covers all families.
    assert out["filtering_method"] == "generalized_gaussian_fisher"


def test_categorical_filter_returns_psd_covariance():
    def obs(x, ph, u):
        # Logits; families.py applies the softmax.
        return jnp.asarray([x[0], 0.0, -x[0]])

    model = StateModel(
        lambda x, th, u, y: x, obs, "categorical", Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])), [0.2], initial_state_covariance=[0.7]
    )
    out = model.evaluate([0, 0], {"y": np.array([2]), "u": None})
    assert out["state_covariance"].shape == (1, 1, 1)
    assert out["state_covariance"][0, 0, 0] >= 0
    assert np.isfinite(out["loglik"]).all()


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "categorical"])
@pytest.mark.parametrize("covariance", [np.eye(2), np.diag([1.0, 0.0])])
def test_multistate_isotropic_and_singular_support_has_finite_ad(family, covariance):
    def observation(x, phi, u):
        if family == "gaussian":
            return x
        if family == "bernoulli":
            return x[0] + x[1]
        return jnp.array([x[0], x[1], 0.0])

    model = StateModel(
        evolution=lambda x, th, u, y: th[0] * x,
        observation=observation,
        family=family,
        priors=Priors(GaussianPrior([0.8], [0.1]), GaussianPrior.empty()),
        initial_state=[0.0, 0.0],
        initial_state_covariance=covariance,
        process_covariance=0.1 * covariance,
        observation_covariance=0.2 if family == "gaussian" else None,
    )
    y = np.array([[0.1, -0.2], [0.4, 0.2], [0.0, 0.3]]) if family == "gaussian" else np.array([0, 1, 0])
    data = prepare_subject_data({"y": y, "u": None})
    f = lambda p: jnp.sum(model.evaluate_jax(p, data, filter_jitter=0.0)["loglik"])
    p = jnp.asarray([0.8])
    g = np.asarray(jax.grad(f)(p))
    H = np.asarray(jax.hessian(f)(p))
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(H))
    h = 1e-4
    fd1 = (float(f(p + h)) - float(f(p - h))) / (2 * h)
    fd2 = (float(f(p + h)) - 2 * float(f(p)) + float(f(p - h))) / (h * h)
    np.testing.assert_allclose(g, [fd1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(H, [[fd2]], rtol=1e-4, atol=2e-5)
    run = model.evaluate([0.8], {"y": y, "u": None})
    if covariance[1, 1] == 0:
        np.testing.assert_allclose(run["state_covariance"][:, 1, :], 0, atol=1e-12)


def test_nonlinear_gaussian_filter_matches_ordinary_ekf():
    model = StateModel(
        observation=lambda x, phi, u: x[0] ** 2,
        priors=Priors(observation=GaussianPrior.empty()),
        initial_state=[1.0],
        initial_state_covariance=0.5,
        observation_covariance=0.2,
    )
    out = model.evaluate([], {"y": np.array([4.0]), "u": None})
    K = 1 / 2.2
    assert out["states"][0, 0] == pytest.approx(1 + K * 3, abs=1e-12)
    assert out["state_covariance"][0, 0, 0] == pytest.approx(0.5 * (1 - 2 * K), abs=1e-12)
    assert out["loglik"][0] == pytest.approx(norm.logpdf(4, 1, np.sqrt(2.2)), abs=1e-12)
    assert out["update_converged"][0]


def test_standalone_evaluation_uses_the_fitting_filter_defaults():
    from gbmtoolbox import Config
    from gbmtoolbox.state_model import filter_settings

    assert filter_settings() == filter_settings(Config(verbose=False))
