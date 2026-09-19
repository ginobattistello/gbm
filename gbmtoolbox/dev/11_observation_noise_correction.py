"""Reference check 11: observation-noise correction for parameter uncertainty.

The MAP estimate of the observation SD uses the residuals at the fitted
parameters, which treats those parameters as known exactly. For a linear model
that is the maximum-likelihood variance ``SSE/T``, biased low by a factor
``sqrt((T - p)/T)``. Correcting the residual energy with the expected
prediction uncertainty ``sum_t J_t Sigma J_t^T`` is what recovers the
unbiased scale.

This script checks that claim against the analytic factor, which is known
exactly for a linear model, and reports how the bias varies with the number of
trials and parameters.
"""

import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit, observation_noise_correction

TRUE_SD = 1.0
N_REPLICATES = 60
FLAT_NOISE_PRIOR = 1e6  # isolate the correction from prior shrinkage

#: Tolerance in Monte-Carlo standard errors rather than in absolute SD units.
#: The estimator ``sqrt(SSE/T)`` has standard error ``sigma/sqrt(2(T-p))``, so
#: averaging ``N_REPLICATES`` draws leaves a standard error of
#: ``sigma/sqrt(2(T-p)N)`` -- about 0.02 at T=25, p=5, N=60.  A fixed absolute
#: tolerance tight enough to be meaningful at T=200 would sit below the noise
#: floor at T=25 and fail at random, so the threshold scales with the noise.
TOLERANCE_SIGMAS = 4.0


def monte_carlo_standard_error(n_trials, n_params):
    """Standard error of the mean of ``sqrt(SSE/T)`` over replicates."""
    return TRUE_SD / np.sqrt(2.0 * (n_trials - n_params)) / np.sqrt(N_REPLICATES)


def build_model(n_params):
    keys = "abcd"[: n_params - 1]

    def observation(x, phi, u_t):
        return phi[0] + sum(phi[i + 1] * u_t[k] for i, k in enumerate(keys))

    return StateModel(
        observation=observation,
        family="gaussian",
        priors=Priors(observation=GaussianPrior([0.0] * n_params, [100.0] * n_params, names=[f"b{i}" for i in range(n_params)])),
        initial_state=[0.0],
        observation_noise_prior=GaussianPrior([0.0], [FLAT_NOISE_PRIOR], names=["log_observation_sd"]),
    )


def run_case(n_trials, n_params):
    model = build_model(n_params)
    keys = "abcd"[: n_params - 1]
    residual, corrected = [], []
    for seed in range(N_REPLICATES):
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(n_trials, n_params - 1))
        beta = np.linspace(0.8, -0.5, n_params - 1)
        y = 1.0 + U @ beta + rng.normal(0.0, TRUE_SD, n_trials)
        data = {"y": y, "u": {k: U[:, i] for i, k in enumerate(keys)}}
        fit = individual_fit([data], model, config=Config(num_init=2, random_state=0, display=False))
        c = observation_noise_correction(fit)
        residual.append(c.residual_sd[0])
        corrected.append(c.corrected_sd[0])
    return float(np.mean(residual)), float(np.mean(corrected))


def main():
    print(f"true observation SD = {TRUE_SD}, {N_REPLICATES} replicates per case")
    print(f"errors are reported in Monte-Carlo standard errors (tolerance: {TOLERANCE_SIGMAS} sigma)\n")
    header = f"{'T':>5}{'p':>4}{'residual':>11}{'expected':>11}{'corrected':>11}{'resid z':>9}{'corr z':>9}"
    print(header)
    print("-" * len(header))

    worst_residual_z = 0.0
    worst_corrected_z = 0.0
    for n_trials in (25, 50, 100, 200):
        for n_params in (3, 5):
            residual, corrected = run_case(n_trials, n_params)
            # analytic MLE bias for a linear model with p parameters
            expected_residual = TRUE_SD * np.sqrt((n_trials - n_params) / n_trials)
            se = monte_carlo_standard_error(n_trials, n_params)
            residual_z = abs(residual - expected_residual) / se
            corrected_z = abs(corrected - TRUE_SD) / se
            worst_residual_z = max(worst_residual_z, residual_z)
            worst_corrected_z = max(worst_corrected_z, corrected_z)
            print(f"{n_trials:>5}{n_params:>4}{residual:>11.4f}{expected_residual:>11.4f}{corrected:>11.4f}{residual_z:>9.2f}{corrected_z:>9.2f}")

    print(f"\nworst residual deviation from the analytic MLE bias : {worst_residual_z:.2f} sigma")
    print(f"worst corrected deviation from the true SD          : {worst_corrected_z:.2f} sigma")
    print(
        "\nThe uncorrected estimate reproduces the analytic MLE bias, confirming that\n"
        "the MAP noise estimate treats the fitted parameters as known exactly.\n"
        "The correction removes most of that bias. It is applied in a single step\n"
        "rather than iterated to self-consistency, so a small downward bias\n"
        "survives at the smallest trial counts; it shrinks as T grows."
    )

    if worst_residual_z < TOLERANCE_SIGMAS and worst_corrected_z < TOLERANCE_SIGMAS:
        print("PASS")
    else:
        print("FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
