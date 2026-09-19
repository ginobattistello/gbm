"""Tutorial 09 — observation-only models (no evolution) for all three families.

Some models have no latent dynamics at all: the outcome on each trial depends
only on that trial's inputs. For these, omit ``evolution`` from the StateModel
and omit the evolution prior from ``Priors``; ``theta`` is then an empty vector.

``initial_state`` can be omitted too. It defaults to ``[0.0]`` and, with no
evolution and an observation function that ignores ``x``, its value makes no
difference to the fit -- it is only there because the trial recursion needs a
state to carry.

This script also shows two things that are easy to get wrong:

1. ``observation`` returns the *natural parameter* -- a logit for Bernoulli and
   a vector of logits for categorical -- never a probability. The toolbox
   applies the sigmoid/softmax itself when it builds the likelihood.
2. The fitted *probabilities* are still reported, in ``output.prediction``.
"""

import jax.numpy as jnp
import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit, observation_noise_prior_from_scale

rng = np.random.default_rng(0)
n_trials = 200
u = rng.normal(size=n_trials)

config = Config(num_init=4, random_state=0, display=False)

# --------------------------------------------------------------- continuous
# No evolution function and no evolution prior: an observation-only model.
y_continuous = 0.5 + 1.2 * u + rng.normal(0.0, 0.6, n_trials)

continuous = StateModel(
    observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t,  # Gaussian mean
    family="gaussian",
    priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["intercept", "slope"])),
    # Centre the noise prior on the scale of the data instead of assuming
    # SD ~ 1. Worth doing whenever the outcome is not already O(1), and
    # especially with few trials.
    observation_noise_prior=observation_noise_prior_from_scale(y_continuous),
    name="observation-only gaussian",
)
print("continuous parameters:", continuous.parameter_layout.names)

fit = individual_fit([{"y": y_continuous, "u": u}], continuous, config=config)
print(fit.summary(0))
print("theta is empty:", fit.output.evolution_parameters.shape, "\n")

# ------------------------------------------------------------------- binary
p_true = 1.0 / (1.0 + np.exp(-(0.3 + 1.5 * u)))
y_binary = (rng.random(n_trials) < p_true).astype(int)

binary = StateModel(
    observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t,  # LOGIT, not p
    family="bernoulli",
    priors=Priors(observation=GaussianPrior([0.0, 0.0], [4.0, 4.0], names=["b0", "b1"])),
    name="observation-only bernoulli",
)
fit_binary = individual_fit([{"y": y_binary, "u": u}], binary, config=config)
print("binary estimates:", np.round(fit_binary.output.observation_parameters[0], 3))

# The observation function returns logits, but the fitted probabilities are
# reported here -- p(y=1) on every trial.
p_hat = fit_binary.output.prediction[0]
print("fitted p(y=1), first 5:", np.round(p_hat[:5], 3))
print("correlation with the true probabilities:", round(float(np.corrcoef(p_hat, p_true)[0, 1]), 4), "\n")

# -------------------------------------------------------------- categorical
# Three classes; class 0 is the reference and its logit is fixed to 0, which
# is what makes the remaining parameters identifiable.
logits_true = np.stack([np.zeros(n_trials), 0.5 + 1.0 * u, -0.4 - 0.8 * u], axis=1)
probs_true = np.exp(logits_true - logits_true.max(axis=1, keepdims=True))
probs_true /= probs_true.sum(axis=1, keepdims=True)
y_categorical = np.array([rng.choice(3, p=row) for row in probs_true])

categorical = StateModel(
    observation=lambda x, phi, u_t: jnp.array(
        [0.0, phi[0] + phi[1] * u_t, phi[2] + phi[3] * u_t]  # vector of LOGITS
    ),
    family="categorical",
    priors=Priors(observation=GaussianPrior([0.0] * 4, [4.0] * 4, names=["a1", "b1", "a2", "b2"])),
    observation_dim=3,
    name="observation-only categorical",
)
fit_categorical = individual_fit([{"y": y_categorical, "u": u}], categorical, config=config)
print("categorical estimates:", np.round(fit_categorical.output.observation_parameters[0], 3))

# One probability per class per trial; each row sums to 1.
probs_hat = fit_categorical.output.prediction[0]
print("fitted class probabilities, first 3 trials:\n", np.round(probs_hat[:3], 3))
print("row sums:", np.round(probs_hat[:3].sum(axis=1), 6))
