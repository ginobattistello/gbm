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
    alpha = jax.nn.sigmoid(theta[0])  # learning rate in (0, 1)
    choice = y_t.astype(jnp.int32)
    reward = u_t["reward"]
    return x.at[choice].add(alpha * (reward - x[choice]))


def observation(x, phi, u_t):
    beta = jnp.exp(phi[0])  # choice temperature > 0
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

fit = individual_fit(data, model, config=Config(latent_uncertainty="propagated"))
```

This is the Rescorla-Wagner update,
$Q_{c} \leftarrow Q_{c} + \alpha(r - Q_{c})$, applied to the chosen option.

`data` is a list with one dict per subject, `{"y": outcomes, "u": inputs}`,
where every array has trials on its first axis and `u` is optional. For one
subject, the outcome vector can be passed directly, with inputs as a keyword:
`individual_fit(y, model, u=u)`. `u` may be an array of any dimension (the
model receives `u[t]`), a dict of such arrays, or omitted. Both forms run the
same fit, and `fit.data[0]` stores the subject as a dict either way. A stacked
subjects-by-trials matrix is not accepted, because its rows could equally be
trials of a multivariate outcome.

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

## Models with no latent dynamics

Not every model has a state that evolves. When the outcome on each trial
depends only on that trial's inputs – a psychometric function, a regression on
stimulus features – omit the evolution function and the evolution prior:

```python
model = StateModel(
    observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t,
    family="gaussian",
    priors=Priors(observation=GaussianPrior([0.0, 0.0], [1.0, 1.0], names=["b0", "b1"])),
)
```

$\theta$ is then an empty vector and only the observation parameters are
estimated. `initial_state` may be omitted too: it defaults to `[0.0]`, and
with no evolution the state never changes, so an observation function that
ignores $x$ gives the same fit whatever it holds. Independent deterministic
trials are evaluated with `jax.vmap`, and the constant state is broadcast for
reporting. Set it when the observation function reads $x$. A stochastic state
still requires filtering even when no evolution function is supplied.

Declaring an evolution prior without an evolution function is an error, since
that would announce parameters nothing can use.

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

set through `observation_covariance`. Bernoulli and categorical outcomes take
no extra $R_t$: their noise is already in the outcome distribution, and passing
`observation_covariance` for them is an error.

**Gaussian observation noise is estimated by default.** A Gaussian model has no
likelihood at all without $R_t$ — the predictor $g$ is a point, and a point
assigns zero density to a continuous outcome. So omitting `observation_covariance`
does not mean "no observation noise"; it means *estimate it*. The toolbox then
appends `log_observation_sd` to the parameter vector and infers
$R = \mathrm{diag}(e^{2\rho})$ from the data, under a default prior on $\rho$.
The default is **exactly** the proper Gamma prior on precision used by VBA:

$$\lambda_j=\sigma_j^{-2}\sim\operatorname{Gamma}(a=1,b=1),$$

where $b$ is the **rate**, not the scale. It is fixed before fitting; no
subject-specific outcome scale is read to construct it. No observation-noise
prior or covariance input is required from the user.

Inference still uses $\rho_j=\log\sigma_j$ in the joint MAP/Laplace vector.
The change of variables is part of the prior density:

$$\log p(\rho_j)=a_j\log b_j-\log\Gamma(a_j)+\log2
-2a_j\rho_j-b_j e^{-2\rho_j}.$$

The Jacobian $|d\lambda/d\rho|=2e^{-2\rho}$ and all normalizing constants
are included once in the prior. Automatic differentiation of the complete
log joint supplies optimization gradients and the observed Hessian. For this
prior the negative log-prior curvature is $4b_j e^{-2\rho_j}$, rather than
the inverse of its global variance.

Prior log-SD moments are $E[\rho]=(\log b-\psi(a))/2$ and
$\operatorname{Var}(\rho)=\psi_1(a)/4$. At $a=b=1$, these are approximately
0.2886 and 0.4112. They are summaries and optimizer-proposal moments, not a
Gaussian approximation to the evaluated prior. The exact central 95% SD
interval is approximately $[0.52,6.29]$. At shape one the prior mean of
$\sigma^2$ is infinite; summarize prior noise with quantiles rather than a
mean variance. Prior-predictive simulations sample
Gamma precision directly. Posterior-predictive simulations still sample the
joint Gaussian Laplace posterior.

```python
from gbmtoolbox import GammaPrecisionPrior

StateModel(..., family="gaussian")  # estimate R under exact Gamma(1,1)
StateModel(..., family="gaussian", observation_covariance=0.25)  # fixed variance
StateModel(..., family="gaussian",
           observation_noise_prior=GammaPrecisionPrior(shape=2, rate=0.5))
StateModel(..., family="gaussian",
           observation_noise_prior=GaussianPrior([np.log(0.5)], [0.5]))
```

For multivariate Gaussian outcomes, set `observation_dim` (or supply a matching
vector prior) to estimate one independent precision per observed dimension.
A scalar prior broadcasts independently. Cross-output correlations and
trial-varying noise are not inferred by this diagonal default. VBA assigns
precision hyperparameters by Gaussian source; one source containing several
outputs need not match GBM's independent precisions.

### Noise summaries, scale, and evidence

`fit.output.observation_noise_sd` is $\exp(\widehat\rho)$, the exponentiated
joint log-SD MAP. It is neither the posterior mean nor the SD-coordinate
posterior mode. MAP depends on coordinate. VBA's commonly reported
$\sqrt{b_{\mathrm{post}}/a_{\mathrm{post}}}$ is
$1/\sqrt{E[\lambda\mid y]}$, not $E[\sigma\mid y]$.

The matching Gamma prior does not reproduce VBA's inference algorithm. GBM
uses joint MAP and observed-Hessian Laplace inference; VBA uses variational
updates and free energy. Differences can remain for fitted scientific
parameters, uncertainty, and log evidence. Stochastic GBM models additionally
use their existing filtering approximation; the conjugate reference below
applies only when predictions are fixed and the state is deterministic.

For fixed scalar predictions, with $S=\sum_t(y_t-\mu_t)^2$,

$$A=a+T/2,\quad B=b+S/2,\quad
\widehat\rho=\tfrac12\log(B/A),\quad H_{\rho\rho}=4A.$$

The precision posterior is Gamma($A,B$). Its integral gives the exact evidence
reference

$$\log Z=-\tfrac T2\log(2\pi)+a\log b-\log\Gamma(a)
+\log\Gamma(A)-A\log B.$$

The toolbox continues to report **Laplace evidence**, even in this reference
case; it does not substitute the analytic integral. In this one-dimensional
case, its log-evidence error at the exact mode is

$$\log Z_{\mathrm{Laplace}}-\log Z
=(A-\tfrac12)\log A-A+\tfrac12\log(2\pi)-\log\Gamma(A).$$

The error approaches zero with more observations. Numerical optimizer error
adds a small residual. The Gamma prior is proper even at zero residual energy
because $b>0$; no additional noise floor is introduced.

**Automatic does not mean scale invariant.** The default Gamma(1,1) fixes a
reference scale in outcome units. It is not the improper Jeffreys prior.
Under $y'=cy$, preserve the same model by transforming predictions, affected
scientific parameter priors, and the Gamma rate: $b'=c^2b$. Exact log evidence
then shifts by $-N\log|c|$, with $N$ scalar observations, a common shift that
cancels in Bayes factors on the same data. Laplace evidence follows this rule
for a consistently transformed parameterization, up to numerical error.
Keeping Gamma(1,1) after changing units specifies different prior beliefs.

`observation_noise_prior_from_scale(y, prior_sample_size=nu0)` returns the
same exact form, $\lambda\sim\operatorname{Gamma}(\nu_0/2,\ \nu_0 s_0^2/2)$,
so that $E[\lambda]=1/s_0^2$; `prior_scale=1` with $\nu_0=2$ is the default
prior. Passing observed `y` uses its MAD (with an SD fallback) to choose
$s_0$ and therefore gives empirical-Bayes evidence conditional on that scale;
an outcome with no scale information (constant, or one trial) raises and asks
for `prior_scale`. It is not used by the
automatic default. Do not derive noise-prior scales from model-specific fitted
residuals for model comparison.

### Compatibility and model-comparison impact

| Model or operation | Effect of the exact default |
|---|---|
| Deterministic Gaussian, omitted noise prior/covariance | MAP, covariance, noise estimates, and log evidence can change |
| Stochastic Gaussian, omitted noise prior/covariance | Same prior change; filtering equations are unchanged but fitted parameters and filtered predictions can change |
| Gaussian with explicit Gaussian log-SD prior | Same fitting objective, Hessian, and evidence as before |
| Gaussian with fixed or callable covariance | No new precision parameter or prior; fitting objective unchanged |
| Bernoulli or categorical | No observation-precision prior is added; fitting objective and filtering unchanged |
| BMS with the same evidence matrix | Algorithm and results unchanged |
| BMS supplied newly fitted Gaussian evidence | Responsibilities, frequencies, and protected exceedance probabilities can change |
| HBI | Group updates unchanged; exact noise prior preserved, but changed Gaussian subject fits/evidence can change group results |
| Prior predictive | Exact Gamma draws for the new default; explicit Gaussian and discrete prior draws retain their existing seeded behavior |
| Information diagnostic | Subtracts actual prior curvature at the MAP, then whitens using global prior moments |

Do not mix historical and new default Gaussian evidence values. Refit all
candidate models for a consistent comparison. This is a Bayesian change of
prior, not an empirical-Bayes change or an extra model-selection penalty.
Evidence comparisons require models for the same observed data and observation
measure; Gaussian densities and binary/categorical masses for differently
encoded outcomes are not directly comparable.

The earlier Gaussian approximation can still be stated explicitly as a
`GaussianPrior` on log SD with the moments above, mean
$(\log b-\psi(a))/2$ and variance $\psi_1(a)/4$.

`fit.input.observation_noise_prior` records the resolved distribution and its
parameters. `prior_mean` and `prior_covariance` remain moments in the fitted
coordinates and do not alone specify a non-Gaussian density.

The distinction matters because it changes the likelihood, not just the
reporting. If the state is uncertain, it must be integrated out (below). If it
is deterministic, there is nothing to integrate.

## Estimation

### Priors and the posterior

Evolution/observation parameters get Gaussian priors; estimated
observation noise uses the exact default above unless overridden. Setting a prior variance to zero
**fixes** that parameter at its prior mean: it is removed from the optimisation
and reported with zero posterior variance. This is how you compare a model
against a restricted version of itself.

The toolbox estimates the **maximum a posteriori** value — the parameter vector
maximising the posterior density:

$$\hat{p} = \arg\max_{p}\ \log p(y \mid p) + \log p(p)$$

Search is multi-start L-BFGS-B, from several starting points, keeping the best
optimum found. The default is four starts (`Config(num_init=...)`).
`Config(hard_bounds=...)` takes one `(low, high)` pair or `None` per resolved
parameter, in `model.parameter_layout.names` order, including any estimated
noise log-SDs; either side of a pair may be `None`.

### Fitting performance and optimizer coordinates

Deterministic observation-only models use `jax.vmap`; models with explicit
evolution retain `jax.lax.scan`. Every outcome still contributes its normalized
likelihood. Scalar and diagonal Gaussian covariances use elementwise arithmetic,
while full or callable covariance specifications retain the general matrix
likelihood. No filtering approximation, prior density, or evidence formula is
changed by these execution paths.

Parameter scaling is disabled by default; enable it with
`Config(parameter_scaling=True)`. In free parameter space the
optimizer works with $x=\mu+Bz$. Without hard bounds, $B$ is the Cholesky factor
of the free prior covariance. With bounds, $B$ is diagonal with the marginal
prior standard deviations, so each original box constraint maps to a box
constraint in $z$. Fixed parameters are removed before scaling. For exact Gamma
noise, prior log-SD moments define only the numerical scale; the objective still
uses the normalized exact Gamma density and its transformation Jacobian.

The objective is the original log joint evaluated at $x=\mu+Bz$. No coordinate
Jacobian is added to the optimization target: this is a change of search
variables, not a different definition of the MAP. If $H_z$ is the observed
Hessian in optimizer coordinates, the returned Hessian is
$H_x=B^{-T}H_zB^{-1}$. Evidence uses $H_x$ and the original normalized log prior.
The finite-difference backend continues to calculate curvature in original
coordinates. Returned gradients, search paths, starts, covariance, and fitted
parameters also use original coordinates.

Scaling changes numerical search paths and the effective stopping criterion.
Nonlinear models can reach a different local optimum. Inspect convergence and
retain multiple starts where needed. With the default
`parameter_scaling=False` the optimizer searches the original free parameters.

Objective and full-evaluation caches use immutable execution settings,
callable identities, parameter block dimensions, and applicable free indices
rather than model identity. Numerical Gaussian and Gamma prior quantities are
dynamic arguments, so HBI and prior-sensitivity refits can reuse compilation.
Changing shapes/dtypes or input dictionary structure can trigger recompilation;
recreating an observation/evolution callable also creates a different cache
entry. Subclasses conservatively use model identity. Callables and captured
settings must remain immutable; construct a new callable/model when those
settings change. Cache entries hold strong references and persist for the
Python process, so reusing model functions limits cache growth.

Data are prepared once per subject. Preflight compiles and checks the same
value-and-gradient executable used by the optimizer, retaining the first
evaluation. It checks the clipped initial point when hard bounds are supplied.
Final reporting uses a cached full evaluator; propagated state uncertainty
uses the same structural caching and prepared data.

Timing fields, in seconds:

| Location | Keys and interpretation |
| --- | --- |
| `fit.profile.timings` | `validation` includes host checks/data preparation; `preflight` includes first objective/gradient compilation and evaluation; `total` includes the whole fit and optional plotting. |
| `fit.math.diagnostics[s].timings` | `first_objective`, `optimization`, `map_components`, `hessian`, `prediction`, `latent_uncertainty`, `optimize_total`, `subject_total`. First-use compilation is included in the relevant stage. |
| `fit.math.diagnostics[s].n_evaluations` | Sum of SciPy objective requests across starts; cached requests are still counted by SciPy. |
| `fit.math.diagnostics[s].starts[k]` | `n_evaluations` and `elapsed_seconds` for one start; `gradient_norm` uses original coordinates. |

Stages containing JAX execution synchronize their results before recording
elapsed time. Totals contain their component stages and should not be added to
them. Subject totals exclude the shared preflight stage; the first subject's
`first_objective` is small because preflight retained that result.

```python
fit = individual_fit(data, model, config=Config(num_init=1, verbose=False))
print(fit.profile.timings)
print(fit.math.diagnostics[0].timings)
print(fit.math.diagnostics[0].n_evaluations)
```

Impact on discrete models and comparison: deterministic Bernoulli/categorical
observation-only models share the vectorized path; their logit definitions and
likelihood normalizations are unchanged. Dynamic and filtered discrete models
retain sequential inference. BMS uses the same original-coordinate Laplace
evidence. HBI can reuse compilation as group priors change, while its statistical
updates remain unchanged. Scaled searches may change which mode is selected;
this numerical effect can affect evidence and should be checked in nonlinear
models. Full posterior curvature is still recomputed at the selected MAP.

Three existing Gamma-prior inconsistencies were also corrected: the documented
`fit.input.observation_noise_prior` metadata is now recorded; prior-predictive
draws use exact Gamma precisions; and `prior_preconditioned_information` subtracts
the exact local Gamma prior curvature ($4b\exp(-2\rho)$), rather than the Gaussian
moment-proxy precision. These change prior-predictive simulations and the
information diagnostic for Gamma noise, but not the fitting density or evidence.
Gaussian-prior predictive sampling and discrete-model densities are unchanged.

### Uncertainty: the Laplace approximation

Around the MAP, the posterior is approximated by the Gaussian that matches its
curvature. Writing $H$ for the observed Hessian of the negative log posterior at
$\hat p$:

$$p(p \mid y) \\approx\ \mathcal{N}\\left(\hat{p},\ H^{-1}\right)$$

Parameter standard errors are the square roots of the diagonal of $H^{-1}$.

Two deliberate choices here. First, **the optimiser's own curvature estimate is
never reused**: L-BFGS-B maintains a quasi-Newton approximation for its own
purposes, and it is not accurate enough for inference. The Hessian is
recomputed from scratch at the final MAP, by automatic differentiation
(`hessian_method="autodiff"`, the default) or central finite differences
(`"central_fd"`). The two agree to finite-difference truncation error.

`map_converged` reports convergence of the selected L-BFGS-B run;
`hess_positive_definite` assesses the independently computed observed Hessian.
`laplace_valid` requires both. If either fails, posterior covariance is `None`
and log evidence is `NaN`. MAP/predictions and diagnostics are retained for
inspection. Converged starts are preferred over unconverged starts; among them,
the largest finite log joint wins. Optimizer success is a numerical criterion,
not a guarantee of a global optimum. Hard-bound solutions remain flagged
`laplace_fragile` because an unconstrained Gaussian may be inappropriate.

Second, **the Hessian is never repaired.** If it is not positive definite, the
Laplace approximation is invalid and is reported as invalid. No eigenvalue
clipping, no nudging toward positive definiteness. A valid MAP and a valid
Laplace approximation are separate outcomes, and a model can legitimately have
the first without the second — which is information about your model, not a
technicality to be smoothed over.

### Model evidence

The same Hessian gives the Laplace approximation to the log marginal
likelihood:

$$\log p(y) \\approx\ \log p(y \mid \hat{p}) + \log p(\hat{p}) + \frac{d}{2}\log 2\pi - \frac{1}{2}\log\det H$$

for $d$ free parameters. The final term penalises models whose fit depends on
finely tuned parameters, which is how the evidence embodies a complexity
penalty without an arbitrary term. This feeds model comparison.

For a stochastic model the first term $\log p(y\mid\hat p)$ is the filtering likelihood, in which the state has already been integrated out (see Filtering). This contributes a second, per-trial complexity penalty in addition to $-\tfrac12\log\det H$. Evidences remain comparable across models, but for Bernoulli and categorical outcomes the inner integral is itself a Laplace approximation, so the evidence inherits that approximation error.

## Filtering: when the state is uncertain

If the state is stochastic, the likelihood of a trial is not conditional on a
single state value. The state must be integrated out:

$$p(y_t \mid y_{1:t-1}, \theta, \phi) = \int p(y_t \mid x_t, \phi)\ p(x_t \mid y_{1:t-1}, \theta, \phi)\ dx_t$$

This is the filtering likelihood, and it is evaluated at *every* parameter
vector the optimiser tries — it is part of estimation, not a post-hoc summary.
How the integral is handled depends on the outcome family:

**Gaussian outcomes** use an extended Kalman update. When the model is actually
linear, this is the exact Kalman filter, and the implementation reproduces the
exact recursion to machine precision with `filter_damping=1` (the default).
The observation Jacobian, innovation covariance, gain, predictive likelihood,
and filtered covariance now all use the same predicted-state linearization.
The covariance uses the Joseph form. Nonlinear Gaussian results can therefore
change relative to the former hybrid update that recomputed curvature at the
updated state. Smaller damping deliberately underweights the mean update and
uses the same damped gain in the Joseph covariance; it is not the ordinary EKF.
Gaussian updates do not require eigenvectors of the state covariance.

**Bernoulli and categorical outcomes** have no closed form. The toolbox uses a
local Laplace-Gaussian update with Fisher scoring: it approximates the
posterior over the state by the Gaussian matching its mode and curvature. This
is genuinely **approximate** Bayesian filtering and is labelled as such
throughout. Discrete updates use $x=m+Ps$ and linear solves involving
$I+PH$; they avoid differentiating covariance eigenvectors, preserve a fixed
semidefinite state support, and remain differentiable at repeated eigenvalues.
Changing support/rank remains a nonsmooth scientific problem. `filter_jitter`
regularizes these solves; it does not add stochastic variance to null directions.
The Fisher curvature differs from the exact observed state Hessian for nonlinear
observation maps. Measured against dense quadrature, the one-step predictive
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
    convergence_diagnostics,  # did multi-start agree?
    posterior_hessian_diagnostics,  # is the Laplace approximation sound?
    prior_preconditioned_information,  # what did the data add over the prior?
    numerical_local_identifiability,  # are parameters separately determined?
)
```

`numerical_local_identifiability` is worth running routinely. If two parameters
only ever appear in combination, it reports a rank deficiency and the direction
that is not determined — a property of your model, visible before you
over-interpret an estimate. It is a numerical check at the fitted point, not a
proof of global identifiability.

For Gaussian models, `observation_noise_correction` asks whether the reported
observation SD can be trusted. That SD comes from the residuals at the fitted
parameters, which treats them as known exactly, so a model with enough freedom
relative to the number of trials leaves residuals that understate the noise --
the familiar gap between $SSE/T$ and $SSE/(T-p)$. The diagnostic recomputes the
noise from the expected residual energy under the Laplace posterior and reports
an inflation factor; near 1 the reported SD is fine, at 1.1 or above it is not.
It changes neither the fit nor its evidence. It applies to deterministic latent
states: with initial-state or process noise the residuals at the predicted
state also contain latent-state variance, so it returns `valid=False` there.
The noise correction remains a residual-energy diagnostic: it does not
re-estimate Gamma hyperparameters, replace the noise posterior, or change
model evidence. Its inflation factor compares corrected and uncorrected
residual scales, not the reported MAP SD.
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

### Extended BMS

The extended routine supports model families, paired conditions, factorial
condition layouts, and independent groups. Model evidence is always provided
with subjects as the first axis; indices are zero based.

| Input | Main result | Inferential target |
| --- | --- | --- |
| `(subjects, models)` | `BMSResult` | Population model frequencies and EP/PXP |
| Same matrix with `families` | `BMSResult.families` | Marginal family frequencies and EP/PXP |
| `(subjects, models, conditions)` ndarray | `ConditionBMSResult` | Prevalence of equal versus unequal condition tuples |
| List of group matrices | `GroupBMSResult` | Shared versus separate population frequency profiles |

Explicit `bms_conditions` and `bms_groups` functions are also available. A list
of 2D matrices denotes independent groups; convert paired-condition data to a
3D ndarray instead. Group sample sizes may differ. Each group must have the same
model definitions, model order, and comparable model priors. Paired conditions
must have the same subjects in the same row order. The toolbox can validate
dimensions and finiteness, but cannot verify subject identities or scientific
comparability of the fitted models from an evidence array alone.

#### Families and priors

```python
result = bms(L, families=[[0, 1], [2, 3]],
             family_names=["learning", "no_learning"], random_state=42)
print(result.families.model_frequency)
print(result.families.protected_exceedance_prob)
```

Families must be non-empty, disjoint, and cover each candidate exactly once.
Without families, the historical GBM prior is preserved: $\alpha_{0k}=1$.
With $F$ families and family size $K_f$, the default is
$\alpha_{0k}=1/(FK_f)$ for $k\in f$, so each family has prior mass $1/F$ and
the total prior count is one. Explicit `alpha0` overrides this default.
Prior count affects RFX inference and must be reported. VBA uses one prior
count in total even without families; pass `alpha0=np.full(K, 1 / K)` to
reproduce that.

The default `family_method="marginal"` first integrates a uniformly selected
within-family model separately for each subject:

$$L_{s,f}=\operatorname{logsumexp}_{k\in f}(L_{s,k})-\log K_f.$$

RFX inference then uses these family evidences and prior counts
$\alpha_{0f}=\sum_{k\in f}\alpha_{0k}$. It estimates population **family**
frequencies without estimating population frequencies of models inside each
family. This matches the default family-level between-group comparison.
Duplicating every model in a family leaves its marginal evidence unchanged.

For **two families**, numerical quadrature integrates the frequency posterior:

$$p(r\mid y)\propto r^{a-1}(1-r)^{b-1}
\prod_s[r\exp(L_{s,0})+(1-r)\exp(L_{s,1})].$$

This posterior is generally not Beta/Dirichlet. Means, variances, responsibilities,
evidence, and EP are integrated directly. `families.posterior_parameters` is
therefore `None`; `expected_counts` is the prior plus expected informative-label
counts, **not** parameters of an exact posterior Dirichlet. `frequency_variance`
reports the integrated variance, and `method="marginal_binary_quadrature"`
identifies this calculation. For more than two families, the frequency posterior
and evidence still use mean-field VB (`method="marginal"`).

To retain the previous analysis use `family_method="aggregate"`. It sums the
full model-level variational Dirichlet counts $a_f=\sum_{k\in f}a_k$ and
responsibilities. This is exact aggregation of an **approximate** model posterior,
not exact family inference. With weak evidence and small asymmetric model prior
counts, mean-field updates can produce strongly biased family frequencies.
The aggregate option retains this limitation; use it only with that assumption
explicitly reported.

The model null fixes model frequencies to $1/K$. The family null fixes total
family mass to $1/F$ and model mass inside each family to $1/K_f$:

$$F_{0,f}=\sum_s\log\left[\sum_f\sum_{k\in f}
\frac{\exp(L_{sk})}{FK_f}\right].$$

Family BOR is $\operatorname{sigmoid}(F_{0,f}-F_{1,f})$ under equal hypothesis
prior odds. $F_{1,f}$ is the integrated two-family evidence or the larger-space
variational bound; for aggregate mode it is the full-model bound. Family PXP is
$(1-\mathrm{BOR}_f)\mathrm{EP}_f+\mathrm{BOR}_f/F$.
The null additionally fixes within-family mixtures; it does not integrate an
unknown population model profile within each family. Model BOR and family BOR
remain separate. Use the family BOR to interpret family or condition PXP.

Exactly constant evidence rows contribute an evidence offset but no information
about frequencies. Their labels are marginalized before the VB updates; they do
not increase posterior counts. `n_informative_subjects` records the remaining
rows. This removes the spurious learning from entirely uninformative subjects,
but **does not remove general mean-field bias for weak nonconstant evidence**.
In particular model-only BMS (including two models) retains its historical VB
reports; exact binary integration is used for marginal families and group odds.

`result.families.within[f]` is a separate BMS analysis restricted to family $f$,
with uniform prior counts totaling one. It is conditional on restricting the
candidate space; it is not the same as the full-model posterior renormalized
within a family. Set `within_families=False` to skip these extra fits.
A singleton candidate/family has frequency and EP/PXP one; a structurally
vacuous one-category BOR is reported as one.

#### Paired and factorial conditions

```python
out = bms_conditions(L_conditions, families=[[0, 1], [2, 3]])
print(out.xp, out.pxp)
print(out.best[0].models)
print(out.per_condition[0].families.model_frequency)
```

For $C$ conditions and $K$ models, enumerate $K^C$ model tuples
$t=(m_1,\ldots,m_C)$ and compute
$L_{s,t}=\sum_c L_{s,m_c,c}$. A tuple belongs to `equal` when all its models
are identical, or all belong to the same supplied family. The remaining tuples
belong to `not_equal`. These two tuple families have equal prior mass and total
prior count one, regardless of their different sizes. `alpha0` specifies only
the per-condition analyses; the between-condition hypothesis prior is fixed
as stated above.

`xp` is the EP that equal tuples are more frequent than unequal tuples;
`pxp` protects it using the tuple-family BOR. Neither quantity is a posterior
probability that every subject uses one common model, nor a group-frequency
equality test. `best` identifies the tuple with highest posterior expected
frequency, while `per_condition` retains each condition's standard BMS.
The full arrays are `equal_exceedance_prob`, `equal_protected_exceedance_prob`,
`bor`, `equal_mask`, and `between`. With one condition the function warns and
returns standard BMS. With only one model family, equality is vacuous and its
EP/PXP are one.

Summing log evidences is justified when condition datasets are independent
conditional on the tuple and their condition-specific parameters are separately
integrated with independent priors. Fitting shared parameters across conditions
requires a joint evidence calculation; separate evidences cannot simply be
summed. Overlapping data must not be counted twice.

A factorial layout contains each zero-based condition index once:

```python
out = bms_conditions(L_four_conditions, factors=np.array([[0, 1], [2, 3]]))
print(out.equal_protected_exceedance_prob)  # one result for each axis
```

For each non-singleton axis, equality requires identical model/family labels
along that axis at every fixed combination of the other factor levels. The
original axes are recorded in `factor_axes`; `best[f]` and `between[f]` refer
to the corresponding comparison. This follows VBA's factorial tuple logic.

Enumeration is exponential. `max_tuples=100000` guards allocation and raises
before constructing larger spaces. Increasing it is explicit opt-in; it does
not make a large comparison computationally cheap. Above 10,000 tuples a warning
is issued. Family aggregation saves EP sampling dimensions, but model-tuple
posteriors and their model EPs still require the full tuple space.

#### Independent groups

```python
out = bms_groups([L_patients, L_controls], families=[[0, 1], [2, 3]])
print(out.probability_equal, out.probability_different)
print(out.log_bayes_factor)  # positive favors different profiles
```

Following the statistical construction in `VBA_groupBMC_btwGroups`, compare
$H_=$ (one population-frequency profile for all groups) with $H_{\ne}$
(independent group-specific profiles). Let $F(L)$ denote the RFX log evidence (integrated for two comparison categories, a variational bound otherwise):

$$F_= = F(\operatorname{stack}_g L_g),\qquad
F_{\ne} = \sum_g F(L_g),\qquad
\log BF_{\ne,=} = F_{\ne}-F_=.$$

With prior equality probability $\pi_=$, the approximate hypothesis posterior is

$$P(H_=\mid y)=\operatorname{sigmoid}
\left[F_=-F_{\ne}+\log\frac{\pi_=}{1-\pi_=}\right].$$

`prior_equal=0.5` is the default. The same Dirichlet prior is used under the
pooled and each separate analysis. This generalizes VBA's two-group test to
two or more independent groups. It compares entire frequency profiles, so
groups can have the same winning model while supporting different profiles.
There are no group model tuples and no group `xp` or `pxp`. For two comparison categories, Gauss-Jacobi quadrature integrates the frequency
likelihood and `evidence_method="binary_quadrature"`. For larger spaces it is
`"variational"`: errors in the two bounds may differ and need not cancel.
Completely uninformative group data preserve `prior_equal` exactly.

Without families, `group_level="auto"` compares model frequencies. With
families it compares family frequencies, using per-subject family evidences

$$L_{s,f}=\log\left[\frac{1}{K_f}\sum_{k\in f}\exp(L_{s,k})\right].$$

The `-log(K_f)` normalization prevents large families receiving greater
evidence solely because they contain more models. This specifies uniform model
selection within a family for each subject; it does not learn a population
distribution over within-family models. Family-level Dirichlet prior counts
are sums of the specified/default model prior counts. Use
`group_level="models"` to retain the full model-frequency comparison with
family-balanced priors, as VBA's group function does when given families.
This distinction can change results when groups differ only within a family.

`pooled` and `per_group` retain full model/family reports.
`comparison_pooled` and `comparison_per_group` retain variational comparison
reports even when group odds use quadrature. The actual hypothesis evidences are
`log_evidence_equal` and `log_evidence_different`; the separate
`variational_log_evidence_*` fields retain the bounds for inspection. `group_sizes`, `group_names`,
`comparison_level`, and `prior_equal` record the analysis choices. Irrelevant
subject-specific evidence constants are removed internally when computing
BOR and group Bayes factors, while reported evidence values retain the original
constants.

#### Exceedance sampling and differences from VBA

Variational Dirichlet EPs for two candidates follow an analytic Beta tail; a single candidate has
EP one. Larger spaces use seeded Dirichlet draws in blocks bounded to about
two million entries. Expected frequencies are analytic. For sampled EPs the
Monte Carlo standard error is at most $0.5/\sqrt{n_{samples}}$ per entry;
the default 100,000 draws gives an upper bound about 0.0016. There is no
full draws-by-model sample allocation for family aggregation.

```python
from gbmtoolbox import bms

conditions = bms(L_conditions, families=[[0, 1], [2, 3]])  # subjects x models x conditions
print(conditions.pxp)
groups = bms([L_patients, L_controls], families=[[0, 1], [2, 3]])  # one matrix per group
print(groups.probability_equal)
```

Differences from VBA's `VBA_groupBMC`, `VBA_groupBMC_btwConds` and
`VBA_groupBMC_btwGroups`:

- Families default to uniform within-family evidence marginalization. Two-family
  reports integrate the frequency posterior; use `family_method="aggregate"`
  to recover the previous variational Dirichlet aggregation.
- The family-BOR constant adjustment is removed. The explicit fixed family
  null is used instead. VBA's source computes its family null with an entropy
  expression based on unweighted model attributions; GBM evaluates the stated
  fixed-profile mixture exactly. Unequal-family results need not match that
  source line for line.
- Condition BOR is the equal/not_equal family BOR, rather than the model-tuple
  BOR.
- Group tuples are replaced by the pooled/separate frequency test. Group
  results report `probability_equal`/`probability_different`, not `xp`/`pxp`.
  Native family-level group comparison is explicitly different from VBA's
  full model-frequency test with family-balanced priors; select
  `group_level="models"` for the latter.
- Invalid/overlapping/incomplete families, non-finite evidence, inconsistent
  group columns, invalid control values, and excessive tuple counts raise clear
  errors. A single candidate is handled as a degenerate distribution.

Binary EP values use analytic probabilities and therefore may differ slightly
from Monte Carlo estimates even on identical data. These features operate on supplied evidence and do not change model fitting,
noise estimation, parameter-scaling defaults, or HBI parameter updates.

Method references:
[VBA group BMS documentation](https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/),
[family BMS source](https://github.com/MBB-team/VBA-toolbox/blob/master/VBA_groupBMC.m),
[paired-condition source](https://github.com/MBB-team/VBA-toolbox/blob/master/VBA_groupBMC_btwConds.m),
and [between-group source](https://github.com/MBB-team/VBA-toolbox/blob/master/VBA_groupBMC_btwGroups.m).
The extension implements their statistical constructions with the differences
specified above; it is not claimed to be a bit-for-bit MATLAB port.

`hbi_main` implements hierarchical empirical-Bayes refitting: group-level priors
and individual fits are estimated together, so each subject is regularised
toward the group while the group is estimated from the subjects. It updates
model-specific Gaussian group priors by responsibility-weighted posterior moment
matching, using the MAP and independently recomputed observed Hessian for each
refit.

Each sweep checks maximum absolute changes in model frequencies, group-prior
means, and group-prior covariance entries against `HBIConfig.tol`.
`convergence_history` records all three. Empty scientific parameter blocks are
supported. On convergence or exhaustion of `maxiter`, `group_priors` are the
priors actually used for the returned subject fits. The final proposed update
is not applied without a subsequent subject refit. A nonconverged result
remains labeled `converged=False`. This is a numerical consistency fix, not a
full variational HBI algorithm. Every subject must have a converged optimizer
run and positive-definite observed Hessian; otherwise HBI raises an error.

### Differences from the original HBI

This is an HBI-*inspired* implementation, not a reproduction of the variational
equations in Piray et al. (2019) or of the reference `cbm` code. The iteration
structure is the same — alternate between per-subject MAP/Laplace fits under the
current group prior, responsibility assignment, and a group-prior update — but
the following differ deliberately:

- **Empirical Bayes rather than full variational Bayes.** The original places a
  Normal-Gamma hyperprior on the group mean and precision and updates its
  variational parameters (`a`, `beta`, `nu`, `sigma`). Here the group prior is
  re-estimated directly by moment matching from the weighted subject posteriors,
  with no hyperprior and no shrinkage toward a prior group mean. Consequently
  there is no `group_hierarchical_errorbar`: uncertainty on the group mean is
  not propagated.
- **Full covariance instead of diagonal.** The original tracks only the diagonal
  of the inverse Hessian and assumes a diagonal group precision. Here the group
  covariance is a full matrix, built from each subject's complete Laplace block
  plus the between-subject spread, then symmetrised and eigenvalue-floored to
  stay positive definite.
- **Responsibilities from Laplace evidence.** Responsibilities combine the full
  Laplace log evidence with `psi(alpha) - psi(sum(alpha))`. The original instead
  uses `logf - 0.5*log|A|` corrected by expected-precision terms arising from the
  variational bound.
- **No free-energy bound.** The original monitors the variational lower bound and
  reports exceedance and protected exceedance probabilities. Convergence here is
  assessed on changes in model frequency, group means, and group covariance, and no bound is
  computed; use `bms` for exceedance probabilities.
- **Scope.** Group refitting applies to evolution (`theta`) and observation
  (`phi`) parameters only. Process- and observation-noise priors stay at their
  model-level specification, though noise parameters are still fitted per subject
  and enter each subject's Hessian and evidence. Fixed parameters are restored
  exactly after each update.


## Reading the result

```python
fit.output.parameters  # MAP estimates, subjects x parameters
fit.output.log_evidence  # Laplace log evidence per subject
fit.output.latent  # state trajectories and uncertainty
fit.math.covariance  # posterior covariance, or None if Laplace invalid
fit.math.hessian  # observed Hessian at the MAP
fit.math.diagnostics  # per-subject convergence and Laplace status
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
9. `09_observation_only.py` — models with no latent dynamics, all three families
10. `10_observation_noise_correction.py` — is the reported observation noise trustworthy?
11. `11_family_condition_group_bms.py` — family, paired-condition and between-group model selection

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




## Reproducibility and variable runtime

`Config.random_state` controls optimization starts and propagated latent draws.
Each subject now has separate optimization and uncertainty streams, so changing
`latent_samples` or requesting propagated reporting cannot change later subjects'
starts. Appending subjects preserves earlier streams; reordering subjects
changes which stream is assigned to which subject. The same seed produces a
new sequence compared with older GBM releases. Nonlinear fits can consequently
select a different mode. Recorded `starts` permit direct comparison.

When `hbi_config` is omitted, HBI inherits `fit_config.random_state`. An explicit
`HBIConfig.random_state` takes precedence. Model-specific seeds are stored in
`HBIResult.model_seeds` and reused across sweeps. `None` deliberately requests
unseeded draws. BMS EP sampling uses `bms(..., random_state=...)`; prior/posterior
predictive helpers use their own `random_state` arguments. These do not inherit
an unrelated fit configuration. Two-family quadrature/group odds have no RNG.
Data generation and random operations inside user functions remain the user's
responsibility: evolution/observation functions must be deterministic for fixed
arguments during fitting.

A seed does not fix runtime. The first JAX call may compile; later calls can
reuse in-memory or persistent compiled executables. Unless a JAX compilation
cache directory is already configured (`jax.config` or
`JAX_COMPILATION_CACHE_DIR`), importing the toolbox sets one at
`~/.cache/gbmtoolbox/jax`; disable it with
`jax.config.update("jax_enable_compilation_cache", False)`. Model functions, shapes, dtypes,
filter settings, process/device initialization, hardware load, and numerical
thread scheduling also affect timing. Separate cold-process timing from repeated
warm fits. Use `fit.math.diagnostics[s].timings`, `n_evaluations`, and start
records to distinguish setup time from different optimizer work. Host result
conversion synchronizes GBM's recorded inference stages; hand-written JAX timers
should call `block_until_ready()` before stopping. Numerical results across
JAX versions or different hardware are not promised to be bitwise identical.
See [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) and
[asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html).
