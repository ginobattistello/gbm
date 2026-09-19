import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from gbmtoolbox import GaussianPrior, Priors, StateModel


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
