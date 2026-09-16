# Manual

A guide to fitting generative models of behaviour: what the toolbox estimates,
the mathematics behind it, and how to express a model in code.

## What this toolbox is for

You have a theory about how someone solves a task — how they learn from
feedback, accumulate evidence, or track a changing world. The theory says there
is something internal that changes trial by trial, and that what you observe on
each trial depends on that internal quantity. You want to know what values of
the theory's parameters best explain a participant's data, how certain you can
be about those values, and whether one theory explains the data better than
another.

That is what this toolbox does. You supply two functions describing your
theory; it supplies the estimation machinery: parameter estimates with
uncertainty, latent-state trajectories, model evidence, diagnostics, and group
comparison.

## The model

Everything is written as a generative state model, which separates three
things:

| symbol | meaning |
|---|---|
| $x_t$ | the **latent state** — what changes from trial to trial |
| $\theta$ | **evolution parameters** — static, govern how the state changes |
| $\phi$ | **observation parameters** — static, govern how the state produces behaviour |

For a reinforcement-learning model, $x_t$ might be the values of the available
options, $\theta$ the learning rate, and $\phi$ the choice temperature.

You write two functions:

```python
def evolution(x, theta, u_t, y_t):
    """Latent state x_t -> x_{t+1}."""

def observation(x, phi, u_t):
    """Latent state x_t -> what is predicted on this trial."""
```

`u_t` is whatever the trial presents — stimuli, rewards, condition codes. It
carries no reserved field names; the toolbox never requires a field called
`reward` or `stimulus`. `y_t` is the outcome observed on the current trial.

Each trial runs in this order:

```text
x_t -> observation -> likelihood of y_t -> evolution -> x_{t+1}
```

The state is used to predict the trial's outcome *before* it is updated with
that outcome, so nothing predicts itself.

### What `observation` must return

This is the detail people get wrong most often. The observation function
returns the **natural parameter** of the outcome distribution, not a
probability. The toolbox applies the link function itself.

| family | return | not |
|---|---|---|
| Gaussian | predicted mean $\mu_t$ | — |
| Bernoulli | the **logit**, $\log\frac{p}{1-p}$ | not $p$ |
| categorical | a vector of **logits** | not a probability vector |

So a softmax choice rule returns $\beta(Q_1 - Q_0)$, not
$\sigma\!\left(\beta(Q_1 - Q_0)\right)$. Returning a probability applies the
sigmoid twice: the model still runs and still fits, but it is not the model you
wrote. Logits are also numerically better behaved, which is why the toolbox
works in that parameterisation.

### A complete example

A two-option learning model with a softmax choice rule:

```python
import jax
import jax.numpy as jnp
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def evolution(x, theta, u_t, y_t):
    alpha = jax.nn.sigmoid(theta[0])        # learning rate in (0, 1)
    choice = y_t.astype(jnp.int32)
    reward = u_t["reward"]
    return x.at[choice].add(alpha * (reward - x[choice]))


def observation(x, phi, u_t):
    beta = jnp.exp(phi[0])                  # choice temperature > 0
    return beta * (x[1] - x[0])             # Bernoulli logit


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

fit = individual_fit(data, model, config=Config(latent_uncertainty="propagated"))
```

This is the Rescorla-Wagner update,
$Q_{c} \leftarrow Q_{c} + \alpha(r - Q_{c})$, applied to the chosen option.

### Writing model functions

Model functions are compiled and differentiated, which imposes a few rules:

- Use `jnp` operations, not `np`, inside `evolution` and `observation`.
- Arrays are immutable: write `x.at[i].add(v)` or `x.at[i].set(v)`, never
  `x[i] += v`.
- Do not cast traced values with `int()` or `float()`; use
  `y_t.astype(jnp.int32)` for indexing.
- Do not branch in Python on a value that depends on the data; use
  `jnp.where`.
- Keep them deterministic. A random draw inside `evolution` is sampled once and
  frozen into the compiled model, which is rejected at validation.
- Fields in `u` must be numeric or boolean. Encode a condition as `0`/`1`, not
  as `"A"`/`"B"`.

Violations are caught before fitting, with a message naming the cause.

### Transforming parameters

Parameters are estimated on an unbounded scale with Gaussian priors, so
constrained quantities are transformed inside the model function: a learning
rate as `jax.nn.sigmoid(theta[0])`, a positive scale as `jnp.exp(phi[0])`. Name
the raw parameter for what it is (`alpha_raw`, `log_beta`) so output stays
readable.

## Static and dynamic models

A model is **static** — better, deterministic — when the latent state follows
its update rule exactly, with no noise. Given $\theta$ and the data, the whole
trajectory is determined. This is the common case: standard learning models are
deterministic in this sense, even though behaviour is stochastic.

A model is **dynamic**, or stochastic, when the state itself is uncertain:

$$x_{t+1} = f(x_t, \theta, u_t, y_t) + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_t)$$

You get this by setting `process_covariance` (the $Q_t$ above, with dimension
equal to the number of latent states) or `initial_state_covariance`, or both.
Either can be a scalar, a diagonal vector, a full matrix, a callable, or the
string `"diagonal"` to estimate the noise as a parameter.

For Gaussian outcomes there is a second noise term,

$$y_t = g(x_t, \phi, u_t) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, R_t)$$

set through `observation_covariance`, which is required for Gaussian models.
Bernoulli and categorical outcomes take no extra $R_t$: their noise is already
in the outcome distribution.

The distinction matters because it changes the likelihood, not just the
reporting. If the state is uncertain, it must be integrated out (below). If it
is deterministic, there is nothing to integrate.

## Estimation

### Priors and the posterior

Each static parameter gets a Gaussian prior. Setting a prior variance to zero
**fixes** that parameter at its prior mean: it is removed from the optimisation
and reported with zero posterior variance. This is how you compare a model
against a restricted version of itself.

The toolbox estimates the **maximum a posteriori** value — the parameter vector
maximising the posterior density:

$$\hat{p} = \arg\max_{p}\; \log p(y \mid p) + \log p(p)$$

Search is multi-start L-BFGS-B, from several starting points, keeping the best
optimum found. The default is four starts (`Config(num_init=...)`).

### Uncertainty: the Laplace approximation

Around the MAP, the posterior is approximated by the Gaussian that matches its
curvature. Writing $H$ for the observed Hessian of the negative log posterior at
$\hat p$:

$$p(p \mid y) \;\approx\; \mathcal{N}\!\left(\hat{p},\; H^{-1}\right)$$

Parameter standard errors are the square roots of the diagonal of $H^{-1}$.

Two deliberate choices here. First, **the optimiser's own curvature estimate is
never reused**: L-BFGS-B maintains a quasi-Newton approximation for its own
purposes, and it is not accurate enough for inference. The Hessian is
recomputed from scratch at the final MAP, by automatic differentiation
(`hessian_method="autodiff"`, the default) or central finite differences
(`"central_fd"`). The two agree to finite-difference truncation error.

Second, **the Hessian is never repaired.** If it is not positive definite, the
Laplace approximation is invalid and is reported as invalid. No eigenvalue
clipping, no nudging toward positive definiteness. A valid MAP and a valid
Laplace approximation are separate outcomes, and a model can legitimately have
the first without the second — which is information about your model, not a
technicality to be smoothed over.

### Model evidence

The same Hessian gives the Laplace approximation to the log marginal
likelihood:

$$\log p(y) \;\approx\; \log p(y \mid \hat{p}) + \log p(\hat{p}) + \frac{d}{2}\log 2\pi - \frac{1}{2}\log\det H$$

for $d$ free parameters. The final term penalises models whose fit depends on
finely tuned parameters, which is how the evidence embodies a complexity
penalty without an arbitrary term. This feeds model comparison.

## Filtering: when the state is uncertain

If the state is stochastic, the likelihood of a trial is not conditional on a
single state value. The state must be integrated out:

$$p(y_t \mid y_{1:t-1}, \theta, \phi) = \int p(y_t \mid x_t, \phi)\; p(x_t \mid y_{1:t-1}, \theta, \phi)\; dx_t$$

This is the filtering likelihood, and it is evaluated at *every* parameter
vector the optimiser tries — it is part of estimation, not a post-hoc summary.
How the integral is handled depends on the outcome family:

**Gaussian outcomes** use an extended Kalman update. When the model is actually
linear, this is the exact Kalman filter, and the implementation reproduces the
exact recursion to machine precision.

**Bernoulli and categorical outcomes** have no closed form. The toolbox uses a
local Laplace-Gaussian update with Fisher scoring: it approximates the
posterior over the state by the Gaussian matching its mode and curvature. This
is genuinely **approximate** Bayesian filtering and is labelled as such
throughout. Measured against dense quadrature, the one-step predictive
probability is accurate to about `1.2e-3` (Bernoulli) and `4.1e-3`
(categorical) in the documented reference cases. See
[VALIDATION.md](VALIDATION.md).

## Reporting latent uncertainty

`Config(latent_uncertainty=...)` controls what is retained after fitting. It
does **not** make a deterministic model stochastic; if the model has state
noise, filtering happens during estimation regardless of this setting.

`"none"` keeps the state trajectory only.

`"propagated"` answers: *given that I am uncertain about the parameters, how
uncertain am I about the state trajectory?* Parameters are drawn from the
Laplace posterior, the model is rerun for each draw, and the spread across runs
is reported with trial-wise covariance and empirical credible intervals. The
same algorithm serves every family.

`"filtered"` answers a different question: *given the fitted parameters, how
uncertain is the state itself?* This reports the filter's own covariance
$P_t$. It requires a stochastic model — on a deterministic one the answer is
identically zero, so the toolbox rejects the request rather than returning
zeros labelled as uncertainty.

The two are not interchangeable. Propagated uncertainty comes from not knowing
the parameters; filtered uncertainty comes from the state being noisy.

## Checking a fit

Validation runs before optimisation: data shape and outcome support, trial
input lengths, parameter and prior dimensions, name uniqueness, covariance
validity, and a full likelihood-and-gradient evaluation at the prior mean.
Problems surface as errors naming the cause, rather than as a failed fit.

After fitting:

```python
from gbmtoolbox import (
    convergence_diagnostics,          # did multi-start agree?
    posterior_hessian_diagnostics,    # is the Laplace approximation sound?
    prior_preconditioned_information, # what did the data add over the prior?
    numerical_local_identifiability,  # are parameters separately determined?
)
```

`numerical_local_identifiability` is worth running routinely. If two parameters
only ever appear in combination, it reports a rank deficiency and the direction
that is not determined — a property of your model, visible before you
over-interpret an estimate. It is a numerical check at the fitted point, not a
proof of global identifiability.

Also available: `prior_sensitivity`, to see how much conclusions depend on prior
choices, and `prior_predictive` / `posterior_predictive`, to check that the
model generates data resembling what was observed.

## Comparing models

Fit each candidate to every subject, then compare at the group level.

```python
from gbmtoolbox import bms, hbi_main
```

`bms` performs random-effects Bayesian model selection from a
subjects-by-models matrix of log evidences. It treats the model identity as a
random effect: rather than asking which model wins overall, it estimates the
proportion of the population each model best describes, which is robust to a
minority of subjects behaving differently.

`hbi_main` implements hierarchical empirical-Bayes refitting: group-level priors
and individual fits are estimated together, so each subject is regularised
toward the group while the group is estimated from the subjects. It updates
model-specific Gaussian group priors by responsibility-weighted posterior moment
matching, using the MAP and independently recomputed observed Hessian for each
refit. It is an HBI-*inspired* implementation, not a line-by-line reproduction
of the original variational equations.

## Reading the result

```python
fit.output.parameters        # MAP estimates, subjects x parameters
fit.output.log_evidence      # Laplace log evidence per subject
fit.output.latent            # state trajectories and uncertainty
fit.math.covariance          # posterior covariance, or None if Laplace invalid
fit.math.hessian             # observed Hessian at the MAP
fit.math.diagnostics         # per-subject convergence and Laplace status
```

Check `fit.math.diagnostics` before interpreting `covariance`: it is `None`
exactly when the Laplace approximation is invalid, and that is a result worth
knowing rather than an inconvenience.

## Tutorials

Runnable, commented examples in `gbmtoolbox/examples/`:

1. `01_individual_fit.py` — learning model with propagated uncertainty
2. `02_continuous_fit.py` — Gaussian outcomes and observation covariance
3. `03_bms.py` — random-effects model selection
4. `04_hbi.py` — hierarchical empirical-Bayes refitting
5. `05_fixed_parameters.py` — fixing parameters via zero prior variance
6. `06_diagnostics.py` — convergence, Hessian, information, identifiability
7. `07_predictive_checks.py` — prior and posterior predictive checks
8. `08_filtered_state_model.py` — latent uncertainty from filtering

## References

The methodological lineage is the Computational Behavioral Modeling framework
and the HBI method:

> Piray P, Dezfouli A, Heskes T, Frank MJ, Daw ND (2019). Hierarchical Bayesian
> inference for concurrent model fitting and comparison for group studies.
> *PLOS Computational Biology* 15(6): e1007043.
> [doi:10.1371/journal.pcbi.1007043](https://doi.org/10.1371/journal.pcbi.1007043)

Random-effects Bayesian model selection:

> Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009). Bayesian model
> selection for group studies. *NeuroImage* 46(4): 1004–1017.

Background on the Laplace approximation, state-space models and filtering:

> Bishop CM (2006). *Pattern Recognition and Machine Learning*, ch. 4.4 and 13.
> Särkkä S (2013). *Bayesian Filtering and Smoothing*. Cambridge.

For how each numerical claim here is checked, see [VALIDATION.md](VALIDATION.md).
