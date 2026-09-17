"""Default observation-noise prior: recovery against the analytic posterior mode.

Question.  When ``observation_covariance`` is omitted, the toolbox estimates a
diagonal R under a default N(0, 2) prior on ``log_observation_sd``.  Does that
estimate match the value the model implies analytically, and how strongly does
the default prior bend it?

Reference.  For a constant-mean Gaussian model with the mean fixed, the
log posterior in rho = log(sigma) is available in closed form:

    L(rho) = -T*rho - S/(2*exp(2*rho)) - rho^2/(2*v) + const,    S = sum (y-mu)^2

Setting dL/drho = 0 gives the stationary condition

    rho + v*(T - S*exp(-2*rho)) = 0

solved here by Brent's method on an independent grid.  The unpenalised MLE
sigma^2 = S/T is reported alongside, so the prior's influence is visible as the
gap between the two.  No comparison against a previous version of the toolbox
is used.

Criterion.  Relative error above 1e-3 against the analytic posterior mode fails.
"""

import numpy as np
from scipy.optimize import brentq

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit
from gbmtoolbox.parameters import DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE

TRUE_MU = 1.25
V = DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE


def evolution(x, theta, u_t, y_t):
    return x


def observation(x, phi, u_t):
    return TRUE_MU


def analytic_posterior_sd(y, v):
    """Mode of the log-sigma posterior, solved independently of the toolbox."""
    T = y.size
    S = float(np.sum((y - TRUE_MU) ** 2))
    return np.exp(brentq(lambda r: r + v * (T - S * np.exp(-2.0 * r)), -20.0, 20.0)), np.sqrt(S / T)


def run_case(true_sd, T, seed):
    rng = np.random.default_rng(seed)
    y = TRUE_MU + rng.normal(scale=true_sd, size=T)

    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(
            GaussianPrior([0.0], [0.0], names=["unused"]),
            GaussianPrior([0.0], [0.0], names=["fixed"]),
        ),
        initial_state=[0.0],
    )
    assert model.observation_covariance_mode == "diagonal", "default did not engage"

    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=3, verbose=False))
    fitted = float(fit.output.observation_noise_sd[0, 0])
    reference, mle = analytic_posterior_sd(y, V)
    return true_sd, T, fitted, reference, mle, abs(fitted - reference) / reference


print(f"default prior variance on log_observation_sd: v = {V}")
print("model: constant mean, all static parameters fixed, R estimated by default\n")
print(f"{'true sd':>8} {'T':>6} {'fitted':>10} {'analytic':>10} {'mle':>10} {'rel err':>10}")

worst = 0.0
for true_sd, T, seed in [(0.3, 200, 1), (0.7, 400, 2), (1.5, 800, 3), (2.5, 200, 4)]:
    true_sd, T, fitted, reference, mle, rel = run_case(true_sd, T, seed)
    worst = max(worst, rel)
    print(f"{true_sd:>8.3f} {T:>6d} {fitted:>10.5f} {reference:>10.5f} {mle:>10.5f} {rel:>10.2e}")

print(f"\nworst relative error against the analytic posterior mode: {worst:.2e}")
print("PASS" if worst < 1e-3 else "FAIL")

# How far does the default prior move the answer away from the pure MLE?
print("\nprior shrinkage (analytic posterior mode vs unpenalised MLE):")
for true_sd, T, seed in [(0.3, 50, 5), (0.3, 200, 5), (0.3, 1000, 5)]:
    rng = np.random.default_rng(seed)
    y = TRUE_MU + rng.normal(scale=true_sd, size=T)
    reference, mle = analytic_posterior_sd(y, V)
    print(f"  T={T:>5d}  posterior sd={reference:.5f}  mle={mle:.5f}  shrinkage={100*(reference-mle)/mle:+.3f}%")
