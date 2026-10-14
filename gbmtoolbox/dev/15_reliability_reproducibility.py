"""Reference check 15: weak-evidence BMS integrals and multistate filter derivatives.

Two-category frequency posteriors under weak or null evidence are compared with
direct Beta-weighted quadrature, and the gradient and Hessian of a two-state
filter likelihood with an isotropic covariance with finite differences.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, expit

from gbmtoolbox import GaussianPrior, Priors, StateModel, bms, bms_groups
from gbmtoolbox.state_model import prepare_subject_data


def integral(L, a=0.5, b=0.5):
    # Independent direct Beta-weighted integration of the frequency likelihood.
    def likelihood(r):
        return np.prod(r * np.exp(L[:, 0]) + (1 - r) * np.exp(L[:, 1]))

    Z = quad(likelihood, 0, 1, weight="alg", wvar=(a - 1, b - 1), epsabs=1e-12)[0]
    mean = quad(lambda r: r * likelihood(r), 0, 1, weight="alg", wvar=(a - 1, b - 1), epsabs=1e-12)[0] / Z
    second = quad(lambda r: r * r * likelihood(r), 0, 1, weight="alg", wvar=(a - 1, b - 1), epsabs=1e-12)[0] / Z
    return np.log(Z) - betaln(a, b), mean, second - mean * mean


def main():
    groups = [np.array([[0.01, 0.0], [0.0, 0.02], [0.015, 0.0]]), np.array([[0.0, 0.01], [0.02, 0.0]])]
    Fe, _, _ = integral(np.vstack(groups))
    Fd = sum(integral(L)[0] for L in groups)
    out = bms_groups(groups, alpha0=[0.5, 0.5], n_samples=1)
    errors = {"weak_group_log_bayes_factor": abs(out.log_bayes_factor - (Fd - Fe)), "weak_group_probability": abs(out.probability_equal - expit(Fe - Fd))}
    L = np.tile([0.001, 0.0], (20, 1))
    F, mean, variance = integral(L)
    families = bms(np.column_stack([L[:, 0], L[:, 1], L[:, 1]]), families=[[0], [1, 2]], family_method="marginal", n_samples=1000).families
    errors.update(
        weak_family_mean=abs(families.model_frequency[0] - mean),
        weak_family_variance=abs(families.frequency_variance[0] - variance),
        weak_family_evidence=abs(families.log_evidence - F),
    )
    null = bms_groups([np.zeros((10, 3)), np.zeros((12, 3))], prior_equal=0.7, n_samples=1000)
    errors["uninformative_group_probability"] = abs(null.probability_equal - 0.7)

    model = StateModel(
        evolution=lambda x, th, u, y: th[0] * x,
        observation=lambda x, phi, u: x,
        family="gaussian",
        priors=Priors(GaussianPrior([0.8], [0.1]), GaussianPrior.empty()),
        initial_state=[0.0, 0.0],
        initial_state_covariance=np.eye(2),
        process_covariance=0.1 * np.eye(2),
        observation_covariance=0.2,
    )
    data = prepare_subject_data({"y": np.array([[0.1, -0.2], [0.4, 0.2], [0.0, 0.3]]), "u": None})
    f = lambda p: jnp.sum(model.evaluate_jax(p, data)["loglik"])
    p = jnp.array([0.8])
    h = 1e-4
    gradient = float(jax.grad(f)(p)[0])
    hessian = float(jax.hessian(f)(p)[0, 0])
    errors["isotropic_gradient_fd"] = abs(gradient - (float(f(p + h)) - float(f(p - h))) / (2 * h))
    errors["isotropic_hessian_fd"] = abs(hessian - (float(f(p + h)) - 2 * float(f(p)) + float(f(p - h))) / (h * h))
    assert all(np.isfinite(v) for v in errors.values())
    assert max(v for name, v in errors.items() if not name.endswith("_fd")) < 1e-10
    assert errors["isotropic_gradient_fd"] < 1e-6
    assert errors["isotropic_hessian_fd"] < 2e-5
    print(json.dumps({"absolute_errors": errors, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
