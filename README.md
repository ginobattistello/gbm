# GBM Toolbox

**Bayesian Generative Brain/Behavior Modelling in Python**

GBM Toolbox is a research toolbox for fitting generative brain/behavior models,
quantifying uncertainty over static parameters and latent states, checking
model adequacy, and comparing models at group level.

GBM Toolbox is scientifically derived from the Computational Behavioral Modeling
(CBM) framework and the HBI method of Piray et al. (2019). The public API and
numerical implementation are redesigned around a single generative state model.
See [NOTICE.md](NOTICE.md).

## Core model

A modeller defines two scientific functions:

```python
def evolution(x, theta, u_t, y_t):
    # latent state x_t -> x_{t+1}
    ...

def observation(x, phi, u_t):
    # latent state x_t -> observation prediction
    ...
```

Notation:

- `x`: dynamic latent state;
- `theta`: static evolution parameters;
- `phi`: static observation parameters;
- `u_t`: arbitrary known trial inputs;
- `y_t`: observed outcome on the current trial.

The trial order is:

```text
x_t -> observation -> likelihood of y_t -> evolution -> x_{t+1}
```

The observation function returns the natural parameter of the outcome
distribution, not a probability:

- Gaussian: predicted mean;
- Bernoulli: the logit, `log(p / (1 - p))`;
- categorical: a vector of logits.

GBM Toolbox applies the sigmoid or softmax itself and constructs the
likelihood. Returning a probability applies the transform twice, which fits a
different model from the one you wrote.

No information is lost by working in logits: the fitted **probabilities** are
reported for every trial in `fit.output.prediction`. For Bernoulli outcomes
that is `p(y = 1)`; for categorical outcomes it is the full probability vector
over classes, with each row summing to 1. Logits are the natural parameter for
the likelihood and keep the optimisation unconstrained and numerically stable,
so they are what the model returns and probabilities are what it reports.

## Minimal example

```python
import jax
import jax.numpy as jnp
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def evolution(x, theta, u_t, y_t):
    alpha = jax.nn.sigmoid(theta[0])
    choice = y_t.astype(jnp.int32)
    reward = u_t["reward"]
    return x.at[choice].add(alpha * (reward - x[choice]))


def observation(x, phi, u_t):
    beta = jnp.exp(phi[0])
    return beta * (x[1] - x[0])  # Bernoulli logit


model = StateModel(
    evolution=evolution,
    observation=observation,
    family="bernoulli",
    priors=Priors(
        evolution=GaussianPrior([0.0], [1.0], names=["alpha_raw"]),
        observation=GaussianPrior([1.0], [1.0], names=["log_beta"]),
    ),
    initial_state=[0.5, 0.5],
    state_names=["Q0", "Q1"],
)

fit = individual_fit(
    data,
    model,
    config=Config(
        latent_uncertainty="propagated",
        display=True,
    ),
)
```

The static parameter dimension is inferred from the priors. A zero prior
variance fixes that static parameter at its prior mean.

## Observation-only models

Some models have no latent dynamics: the outcome depends only on the current
trial's inputs. Omit both the `evolution` function and the evolution prior:

```python
model = StateModel(
    observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t,
    family="gaussian",
    priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])),
)
```

`theta` is then an empty vector. Passing an evolution prior without an evolution
function is an error, since that would declare parameters nothing can use.

`initial_state` can be omitted as well. It defaults to `[0.0]`, and with no
evolution the state never changes, so an observation function that ignores `x`
gives the same fit whatever it holds — the value is inert. It is still carried
because the trial recursion `x_t -> observation -> evolution -> x_{t+1}` needs a
state to thread through; set it only when the observation function actually
reads `x`.

## Data

Each subject is represented as:

```python
data[n] = {
    "y": outcomes,
    "u": inputs,
}
```

`u` is deliberately generic. It may be `None`, a trialwise array, or a
dictionary whose non-scalar values have first dimension equal to the number of
trials. GBM Toolbox never requires field names such as `reward` or `stimulus`.

Field values must be numeric or boolean, so a categorical condition is encoded
as a numeric code (`0`/`1`) rather than a string label.

## Latent uncertainty

Three reporting settings are available:

```python
Config(latent_uncertainty="none")
Config(latent_uncertainty="propagated")
Config(latent_uncertainty="filtered")
```

There are exactly two uncertainty methods:

**Propagated uncertainty** samples static parameters from the final Laplace
posterior and reruns the state model. The same algorithm is used for Gaussian,
Bernoulli, and categorical outcomes. Output includes trial-wise covariance,
standard deviation, and empirical credible intervals.

**Filtered uncertainty** reports uncertainty intrinsic to the latent state,
conditional on fitted static parameters. Gaussian observations use an extended
Kalman filter. Bernoulli and categorical observations use a local
Laplace-Gaussian approximation with Fisher scoring. The latter is approximate
Bayesian filtering and is labelled explicitly as such.

`latent_uncertainty` controls what is retained and displayed. It does not turn
a deterministic model into a stochastic model. If `initial_state_covariance`
or `process_covariance` is non-zero, filtering is part of likelihood evaluation
regardless of the display setting.

## State and observation covariance

For stochastic state evolution:

```text
x_{t+1} = f(x_t, theta, u_t, y_t) + eta_t
eta_t ~ N(0, Q_t)
```

`process_covariance` is `Q_t`; its dimension is the number of latent states.
It can be a scalar, diagonal vector, full covariance matrix, callable, or
`None` for deterministic state evolution.

For Gaussian observations:

```text
y_t = g(x_t, phi, u_t) + epsilon_t
epsilon_t ~ N(0, R_t)
```

`observation_covariance` is `R_t`. For Gaussian models it is **estimated by
default**: omit it and the toolbox infers `R = diag(exp(2 * log_observation_sd))`,
since a Gaussian likelihood does not exist without it. Pass a scalar, vector,
matrix or callable to fix `R` at a known measurement error instead, or supply
`observation_noise_prior` to keep it estimated under your own prior.
Bernoulli/categorical observation families do not use an additional `R_t`.

### The default noise prior

The default is the `Ga(1, 1)` prior on observation **precision** used by the VBA
toolbox, moment-matched onto the log standard deviation that GBM Toolbox
actually fits. It places a 95% prior interval of roughly `[0.38, 4.69]` on the
observation SD.

Two choices are worth spelling out, because they look inconsistent and are not.

**Why a Gamma prior.** A Gamma on precision is the conjugate choice and is what
VBA uses, so taking the same prior makes the two toolboxes agree out of the box
rather than merely in kind.

**Why it is still fitted on the log scale.** GBM Toolbox estimates all static
parameters in one joint MAP/Laplace vector, so the noise posterior is
approximated as Gaussian *in whichever coordinate it is fitted*. That
approximation is much better in `log(sigma)` than in `sigma^2`: for 25 trials
the exact noise posterior has skewness about `-0.28` on the log scale against
`+1.29` for the variance, so the variance scale is roughly five times more
skewed, and the gap is worst exactly where trials are few. VBA can afford to
keep the Gamma form because its variational scheme updates the precision as a
separate conjugate factor and never Laplace-approximates it.

Taking the Gamma prior and fitting on the log scale therefore gets the prior
beliefs of VBA in the coordinate where this toolbox's approximation is most
accurate. Moment matching preserves the first two moments rather than the whole
shape, so the prior is close to VBA's, not identical to it: the exact `Ga(1, 1)`
implies a 95% SD interval of `[0.52, 6.29]` against `[0.38, 4.69]` here.

The default assumes an observation SD of order 1. When the outcome is on a
different scale, state the scale instead, as below.

### Setting the noise prior by scale

The default prior is centred at `log_observation_sd = 0`, i.e. it assumes an
observation SD near 1. Whenever the outcome is on another scale that is a
strong hidden assumption: an SD of 0.05 or 20 sits about two prior standard
deviations from the centre and is shrunk towards 1, and few trials cannot pull
it back.

`observation_noise_prior_from_scale` states the prior the way a conjugate
analysis would — an expected noise scale `s0` and a prior sample size `nu0`:

```python
from gbmtoolbox import observation_noise_prior_from_scale

model = StateModel(
    ...,
    observation_noise_prior=observation_noise_prior_from_scale(subject["y"]),
)
```

This is the Gamma prior `lambda ~ Gamma(nu0/2, nu0*s0**2/2)` on the observation
precision, moment-matched onto `log_observation_sd`. `nu0` contributes that many
pseudo-observations of prior information: the default `nu0=2` gives a 95% prior
interval of roughly `[0.38, 4.69]` times the scale — the same prior as the
default above, recentred — while `nu0=10` is much more informative.

Passing `y` estimates `s0` robustly as `1.4826 * MAD(y)`, which is unaffected by
a few outliers where the standard deviation is not — with 5% gross outliers the
sample SD of unit noise inflates to about 3 while the robust scale stays near 1.
Pass `prior_scale=` instead to set it yourself. Reading the scale from the data
is empirical Bayes, so it is opt-in rather than the default.

Being scale-based, the same model on the same data in different units gives the
same posterior. It removes the bias from assuming an SD near 1; it does not make
the noise better identified, which only more trials can do. Note that a robust
*prior* does not make the *likelihood* robust: gross outliers still enter the
Gaussian likelihood and inflate the fitted SD.

### Is the reported observation noise trustworthy?

The estimated observation SD is driven by the residuals at the fitted
parameters, which treats those parameters as known exactly. A model with enough
freedom relative to the number of trials absorbs part of the noise into its own
parameters, leaving residuals that understate the true observation noise. This
is the familiar difference between the maximum-likelihood variance `SSE/T` and
the unbiased `SSE/(T - p)`, and with 5 free parameters at 25 trials it is a
bias of roughly 10%.

```python
from gbmtoolbox import observation_noise_correction

correction = observation_noise_correction(fit, subject=0)
print(correction.corrected_sd, correction.inflation_factor)
```

This recomputes the noise from the expected residual energy under the Laplace
posterior, adding the prediction uncertainty `sum_t J_t Sigma J_t^T` induced by
parameter uncertainty. An `inflation_factor` near 1 means the reported SD can
be taken at face value; 1.1 or more means it cannot.

It is a **diagnostic**: it changes neither the fit nor its log-evidence, so
model comparison is unaffected. The correction is evaluated once at the MAP
rather than iterated to self-consistency, which removes most of the bias but
leaves a little at the smallest trial counts. See
`dev/11_observation_noise_correction.py` for the measured behaviour against the
analytic result for a linear model.

## MAP and Laplace inference

GBM Toolbox deliberately separates optimizer curvature from Laplace curvature:

```text
multi-start L-BFGS-B
        -> final MAP
        -> independently recomputed observed posterior Hessian (autodiff)
        -> posterior covariance + Laplace evidence
```

The quasi-Newton approximation maintained by L-BFGS-B is never used for
Laplace inference. The final observed Hessian is not repaired by eigenvalue
clipping. A valid MAP may therefore coexist with an invalid Laplace
approximation, which is explicitly flagged.

The Fisher curvature used inside the discrete filter is also distinct from the
outer observed Hessian and is never substituted for it.

## Diagnostics and checks

A centralized preflight validator checks data, outcome support, trial-input
lengths, parameter/prior dimensions, names, state dimensions, covariance
validity, model outputs, and a full prior-mean likelihood evaluation before
optimization.

Candidate-specific numerical failures are recorded separately during
optimization.

Available diagnostics include:

```python
from gbmtoolbox import (
    convergence_diagnostics,
    posterior_hessian_diagnostics,
    prior_preconditioned_information,
    numerical_local_identifiability,
    observation_noise_correction,
)
```

GBM Toolbox also provides prior sensitivity and prior/posterior predictive checks.

## BMS and HBI

```python
from gbmtoolbox import bms, hbi_main
```

`bms` performs random-effects Dirichlet Bayesian model selection from a
subjects-by-models log-evidence matrix.

`hbi_main` currently implements a compact hierarchical empirical-Bayes/Laplace
refitting approximation inspired by the CBM/HBI framework of Piray et al. (2019).
It updates model-specific Gaussian group priors by responsibility-weighted
posterior moment matching and uses GBM Toolbox's MAP plus independently recomputed
observed-Hessian pipeline for each subject refit. It is deliberately labelled as
an HBI-inspired implementation rather than a line-by-line reproduction of the
original HBI variational equations.

## Tutorials

`examples/` contains standalone commented tutorials:

1. `01_individual_fit.py` — deterministic learning model with propagated latent uncertainty;
2. `02_continuous_fit.py` — Gaussian outcomes and observation covariance;
3. `03_bms.py` — random-effects Bayesian model selection;
4. `04_hbi.py` — HBI-inspired hierarchical empirical-Bayes refitting;
5. `05_fixed_parameters.py` — fixed static parameters via zero prior variance;
6. `06_diagnostics.py` — convergence, Hessian, information and local identifiability;
7. `07_predictive_checks.py` — prior and posterior predictive checks;
8. `08_filtered_state_model.py` — internal latent-state uncertainty from filtering;
9. `09_observation_only.py` — observation-only models for all three outcome families;
10. `10_observation_noise_correction.py` — checking whether the reported observation noise is trustworthy.

## Documentation

- [docs/MANUAL.md](docs/MANUAL.md) — what the toolbox estimates, the
  mathematics behind it, and how to write a model.
- [docs/VALIDATION.md](docs/VALIDATION.md) — how the toolbox is checked and
  what the reference cases currently measure.

## Installation

Python 3.10 or newer is required. The inference core is built on JAX, which is
installed automatically as a dependency.

```bash
git clone https://github.com/ginobattistello/BayesGBM.git
cd BayesGBM
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q
```

Run the scientific reference scripts in `gbmtoolbox/dev/` after tests. They are
method-validation scripts, not informal demos.

## Citation and provenance

For the CBM/HBI methodological lineage, cite:

Piray P, Dezfouli A, Heskes T, Frank MJ, Daw ND (2019). *Hierarchical Bayesian
inference for concurrent model fitting and comparison for group studies.*
PLoS Computational Biology. DOI: 10.1371/journal.pcbi.1007043.

For the first GBM Toolbox archival release, update `CITATION.cff`, create a tagged
GitHub release, and archive it with Zenodo.

## Development, AI assistance and contributions

This toolbox was designed and written by its authors with the assistance of AI
coding tools. Every part of the resulting code, mathematics and documentation was
reviewed manually by the authors, and the inference machinery was additionally
validated against independent implementations, in particular the VBA toolbox in
MATLAB, on a real dataset that had already been analysed with it.

AI assistance does not remove the possibility of error, and the authors make no
claim that this review is exhaustive. Contributions of every kind are therefore
warmly encouraged: bug reports, independent replication, method review,
corrections to the mathematics or the documentation, additional validation cases,
and critical reading of the present work. Please open an issue or a pull request.
