"""Tutorial 10 — is the reported observation noise trustworthy?

The observation SD reported by a Gaussian fit is driven by the residuals at the
fitted parameters. That treats those parameters as if they were known exactly,
so a model with enough freedom relative to the number of trials absorbs part of
the noise into its own parameters and reports an observation SD that is too
low. This is the same effect that makes the maximum-likelihood variance
``SSE/T`` biased relative to the unbiased ``SSE/(T - p)``.

``observation_noise_correction`` recomputes the noise from the *expected*
residual energy under the Laplace posterior, which accounts for that parameter
uncertainty. It is a diagnostic: it does not change the fit or its evidence.

Run it whenever you have few trials per subject, many free parameters, or both.
"""

import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit, observation_noise_correction

TRUE_SD = 1.0
N_PARAMETERS = 5
REGRESSORS = "abcd"


def observation(x, phi, u_t):
    return phi[0] + sum(phi[i + 1] * u_t[k] for i, k in enumerate(REGRESSORS))


model = StateModel(
    observation=observation,
    family="gaussian",
    priors=Priors(observation=GaussianPrior([0.0] * N_PARAMETERS, [100.0] * N_PARAMETERS, names=[f"b{i}" for i in range(N_PARAMETERS)])),
    initial_state=[0.0],
)

print(f"true observation SD = {TRUE_SD}, {N_PARAMETERS} free observation parameters\n")
print(f"{'trials':>8}{'reported SD':>14}{'corrected SD':>15}{'inflation':>12}")
print("-" * 49)

for n_trials in (25, 50, 100, 400):
    rng = np.random.default_rng(0)
    U = rng.normal(size=(n_trials, len(REGRESSORS)))
    y = 1.0 + U @ np.array([0.8, -0.5, 0.3, 0.6]) + rng.normal(0.0, TRUE_SD, n_trials)
    data = {"y": y, "u": {k: U[:, i] for i, k in enumerate(REGRESSORS)}}

    fit = individual_fit([data], model, config=Config(num_init=3, random_state=0, display=False))
    correction = observation_noise_correction(fit)

    print(f"{n_trials:>8}{fit.output.observation_noise_sd[0, 0]:>14.4f}{correction.corrected_sd[0]:>15.4f}{correction.inflation_factor[0]:>12.3f}")

print(
    "\nWith few trials the reported SD understates the observation noise, and the\n"
    "inflation factor makes that visible. As the number of trials grows the two\n"
    "converge, because the parameters become well determined and contribute\n"
    "little additional prediction uncertainty.\n"
    "\n"
    "An inflation factor near 1 means the reported SD can be taken at face value.\n"
    "A factor of 1.1 or more means it cannot: the model is flexible enough,\n"
    "relative to the data, that its residuals understate the true noise."
)
