"""Reference check 12: exact Gamma noise prior density, MAP, Hessian, evidence and Bayes factors.

Reference: integrating a Gamma precision analytically with fixed predictions.
Laplace evidence is compared to its known Stirling error, not treated as exact.
Failures raise assertions so this script is suitable for automated validation.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, bms, individual_fit


def fit_fixed(y, mu):
    model = StateModel(observation=lambda x, phi, u: float(mu), priors=Priors(observation=GaussianPrior([0.0], [0.0])))
    return individual_fit([{"y": y, "u": None}], model, config=Config(num_init=2, verbose=False))


def reference(y, mu):
    T = len(y)
    A, B = 1 + T / 2, 1 + np.sum((y - mu) ** 2) / 2
    exact = -T / 2 * np.log(2 * np.pi) + gammaln(A) - A * np.log(B)
    error = (A - 0.5) * np.log(A) - A + 0.5 * np.log(2 * np.pi) - gammaln(A)
    return 0.5 * np.log(B / A), 4 * A, exact, error


def main():
    print("T  rho error       Hessian rel error  Laplace-exact    predicted error")
    rows = []
    for T in (1, 5, 25, 100, 1000):
        y = np.random.default_rng(T).normal(0.2, 0.7, T)
        fit = fit_fixed(y, 0.2)
        rho, H, exact, error = reference(y, 0.2)
        rerr = abs(fit.output.observation_noise_parameters[0, 0] - rho)
        herr = abs(fit.math.hessian[0][0, 0] / H - 1)
        lerr = fit.output.log_evidence[0] - exact
        rows.append((rerr, herr, abs(lerr - error)))
        print(f"{T:4d} {rerr:.3e}       {herr:.3e}          {lerr:+.6e}   {error:+.6e}")
        assert rerr < 2e-6 and herr < 5e-6 and abs(lerr - error) < 2e-6
    zero = fit_fixed(np.zeros(8), 0.0)
    assert np.isfinite(zero.output.log_evidence[0])
    y = np.random.default_rng(15).normal(0.7, 0.4, 20)
    fits = [fit_fixed(y, mu) for mu in (0.0, 0.7)]
    log_bf = fits[0].output.log_evidence[0] - fits[1].output.log_evidence[0]
    exact_bf = reference(y, 0.0)[2] - reference(y, 0.7)[2]
    print(f"log Bayes factor error: {abs(log_bf - exact_bf):.3e}")
    assert abs(log_bf - exact_bf) < 2e-6
    lme = np.array([[f.output.log_evidence[0] for f in fits]])
    r1, r2 = bms(lme), bms(lme + 123)
    bms_error = np.max(np.abs(r1.responsibilities - r2.responsibilities))
    print(f"BMS common evidence shift error: {bms_error:.3e}")
    assert bms_error < 1e-12
    print("max errors (rho, relative Hessian, predicted evidence):", np.max(rows, axis=0))
    y = np.random.default_rng(5).normal(0.3, 0.8, 40)
    model = StateModel(observation=lambda x, phi, u: phi[0], priors=Priors(observation=GaussianPrior([0.0], [1.0])))
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=2, verbose=False))
    A = 1 + len(y) / 2
    C = -len(y) / 2 * np.log(2 * np.pi) + gammaln(A)

    def log_integrand(mu):
        return C - A * np.log(1 + np.sum((y - mu) ** 2) / 2) - 0.5 * (np.log(2 * np.pi) + mu**2)

    offset = log_integrand(fit.output.observation_parameters[0, 0])
    integral, _ = quad(lambda mu: np.exp(log_integrand(mu) - offset), -10, 10, epsabs=1e-11)
    joint_error = fit.output.log_evidence[0] - (offset + np.log(integral))
    print(f"free-intercept joint Laplace evidence minus quadrature: {joint_error:+.6e}")
    assert abs(joint_error) < 0.1
    print("PASS")


if __name__ == "__main__":
    main()
