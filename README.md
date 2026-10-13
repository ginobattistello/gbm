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
gives the same fit whatever it holds. Deterministic observation-only models
evaluate independent trials together using `jax.vmap`; the constant state is
broadcast for reporting. Set it when the observation function actually reads
`x`. Initial-state or process uncertainty still requires filtering.

## Fitting performance

Observation-only models with no state uncertainty use vectorized likelihoods.
Scalar and diagonal Gaussian noise use direct elementwise likelihoods.
Validation reuses the optimizer's compiled value-and-gradient function, data
are prepared once per subject, and final predictions use a cached evaluator.
Compilation is shared when only prior values change, including HBI updates.

Optimization uses prior-scaled coordinates by default. Full Cholesky scaling
is used without hard bounds; diagonal scaling preserves box bounds when they
are supplied. Return values, Hessians, covariance, and evidence remain in the
original parameter coordinates. To compare with the previous search geometry:

```python
config = Config(num_init=1, parameter_scaling=False, verbose=False)
fit = individual_fit(data, model, config=config)
```

For timing, use `fit.profile.timings` (validation, preflight, total) and
`fit.math.diagnostics[subject].timings` (first objective, optimization,
Hessian, predictions, latent uncertainty, and totals). The preflight timer
includes compilation of the first objective/gradient. The Hessian and prediction
timers include their first compilation when needed. `n_evaluations` records
SciPy's objective requests, and each start records its own count and runtime.
Trials with different array shapes can require separate compilations.

See [the manual](docs/MANUAL.md#fitting-performance-and-optimizer-coordinates)
for mathematical details and [validation](docs/VALIDATION.md#performance-update-validation)
for checks and measured before/after timings. Run the reproducible benchmark
with `python gbmtoolbox/dev/13_fit_performance.py`.

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

### Automatic observation noise: exact VBA prior

A deterministic Gaussian model needs no user-supplied observation covariance
or noise prior. Omit both and the toolbox automatically uses the proper prior

$$\lambda_j = \sigma_j^{-2} \sim \operatorname{Gamma}(1,1),$$

where Gamma is parameterized by **shape and rate**. This is the exact default
precision prior used by VBA, rather than the previous Gaussian approximation
to its log-SD moments. Its central 95% observation-SD interval is approximately
`[0.52, 6.29]`, in outcome units.

The fitted coordinate remains `rho = log(sigma)`. The toolbox evaluates the
exact transformed density, including its Jacobian and normalizing constants:

$$\log p(\rho) = a\log b-\log\Gamma(a)+\log 2-2a\rho-b e^{-2\rho}.$$

Noise is estimated jointly with scientific parameters by MAP. Its full
curvature enters the observed Hessian and Laplace log model evidence. The
prior is exact; the posterior and evidence remain Laplace approximations.
VBA uses variational inference, so matching its prior does not imply identical
posterior summaries or evidence values.

```python
model = StateModel(
    observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t,
    family="gaussian",
    priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0])),
)
fit = individual_fit(data, model)  # data is a list of subject dictionaries
print(fit.output.observation_noise_sd)
print(fit.output.log_evidence)
print(fit.input.observation_noise_prior)
```

An omitted `process_covariance` and `initial_state_covariance` keep latent
states deterministic. For multivariate outcomes, set `observation_dim` to
estimate one independent precision per output; a full covariance is not
estimated automatically. VBA groups precisions by Gaussian source, so the
multivariate noise structure must also be matched for comparisons with VBA.

### Optional noise specifications and measurement units

The default is automatic, **not scale invariant**: Gamma(1,1) assumes a noise
scale of order one in the outcome's units. It is fixed before observing data,
so the default evidence does not use a data-derived noise prior.

```python
from gbmtoolbox import GammaPrecisionPrior

# Optional exact prior: shape a and rate b on observation precision.
model = StateModel(..., observation_noise_prior=GammaPrecisionPrior(shape=2, rate=0.5))
# Optional known covariance: the number is variance, not SD.
model = StateModel(..., observation_covariance=0.25)
```

Explicit `GaussianPrior` specifications on log SD remain supported unchanged.
`observation_noise_prior_from_scale(y)` also retains its previous Gaussian
moment approximation. Estimating its scale from the fitted outcomes is an
opt-in empirical-Bayes choice. For strict Bayesian evidence, use the fixed
default or a prior specified from independent information. Keep the same
noise prior across candidate models when that reflects the same measurement
process; do not choose its scale from each candidate's fitted residuals.

Under a unit change `y_new = c * y`, transform predictions and affected
scientific priors consistently and use `rate_new = c**2 * rate`. Log evidence
then changes by the density Jacobian, which cancels between models on the same
data. Resetting the prior to Gamma(1,1) after changing units changes the model.

**Migration:** default Gaussian fits, uncertainty, and evidence can change
relative to the previous moment-matched prior. Refit every candidate Gaussian
model before comparing evidence from this version. Bernoulli/categorical fits,
fixed Gaussian covariance, and explicit Gaussian log-SD priors retain their
existing fitting objectives. See [docs/MANUAL.md](docs/MANUAL.md) for impacts,
legacy reproduction, and [docs/VALIDATION.md](docs/VALIDATION.md) for checks.

### Is the reported observation noise trustworthy?

The observation SD is estimated jointly under its prior and the residual
likelihood. Residuals at the fitted parameters treat those parameters as known
exactly. A model with enough
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
parameter uncertainty. The `inflation_factor` compares two residual-based estimates; it does not
measure the prior's influence or validate the reported posterior noise SD.
Values above 1 indicate prediction uncertainty contributes to residual energy.

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
git clone https://github.com/ginobattistello/gbm.git
cd gbm
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

