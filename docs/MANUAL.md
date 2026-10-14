# GBM manual

GBM fits a generative model to observations. You supply predictions and, when
needed, a state-update rule. It returns parameter estimates, approximate
uncertainty, state trajectories and model evidence.

## Reading guide

**First use:** read sections 1–4, then the result checks in section 8.
**Comparing theories:** also read sections 5, 9 and 10.
**Stochastic states or method review:** read sections 6–8 and 11–12.

1. [Choose a model](#1-choose-a-model)
2. [Write the model and supply data](#2-write-the-model-and-supply-data)
3. [Specify priors and parameter coordinates](#3-specify-priors-and-parameter-coordinates)
4. [Specify state and observation noise](#4-specify-state-and-observation-noise)
5. [Understand MAP, uncertainty and evidence](#5-understand-map-uncertainty-and-evidence)
6. [Understand filtering](#6-understand-filtering)
7. [Read latent uncertainty and predictive simulations](#7-read-latent-uncertainty-and-predictive-simulations)
8. [Check a fit](#8-check-a-fit)
9. [Compare models, families, conditions and groups](#9-compare-models-families-conditions-and-groups)
10. [Refit subjects hierarchically](#10-refit-subjects-hierarchically)
11. [Control speed and reproducibility](#11-control-speed-and-reproducibility)
12. [Report an analysis and find references](#12-report-an-analysis-and-find-references)

## 1. Choose a model

| Your question | Specification | Inference |
|---|---|---|
| Does behaviour depend on current inputs? | Observation function only | Independent-trial likelihood |
| Does behaviour depend on learning/history? | Observation and evolution functions | Deterministic state recursion |
| Is the state itself noisy or initially uncertain? | Also supply initial/process covariance | Filtering inside the likelihood |

“Deterministic state” does not mean deterministic behaviour: outcomes still
follow their observation distribution. A learning model can have changing
states without process noise. Do not confuse deterministic/stochastic with
static/dynamic.

The trial order is **predict the outcome, score the observation, then evolve
the state**. In a stochastic model, the filter first updates the current state
using the observation, then propagates that updated state to the next trial.

| Symbol | Meaning |
|---|---|
| $x_t$ | Latent state on trial $t$ |
| $\theta$, $\phi$ | Evolution and observation parameters |
| $u_t$, $y_t$ | Known inputs and observed outcome |
| $Q_t$, $R_t$ | Process and Gaussian observation covariance |
| $w$ | All free fitted coordinates, including estimated noise log-SDs |

The conditional model is

$$x_1\sim\mathcal N(m_1,P_1),\qquad
x_{t+1}\mid x_t,y_t,u_t,\theta\sim\mathcal N(f(x_t,\theta,u_t,y_t),Q_t).$$

An exactly known initial state or deterministic transition is a point mass.
The likelihood conditions on the supplied inputs and uses the transition and
observation factors in trial order. Noise draws are conditionally independent
under this specification. Inputs must not leak the outcome being predicted;
for example, feedback can enter evolution without being used to predict the
choice that preceded that feedback.

## 2. Write the model and supply data

```python
def evolution(x, theta, u_t, y_t):
    # Return the next state; omit this function for observation-only models.
    ...

def observation(x, phi, u_t):
    # Return the observation predictor below.
    ...
```

| Family | Return from `observation` | Observation model |
|---|---|---|
| `gaussian` | Mean $\mu_t$ | $y_t\sim\mathcal N(\mu_t,R_t)$ |
| `bernoulli` | Logit $\eta_t$ | $P(y_t=1)=\mathrm{sigmoid}(\eta_t)$ |
| `categorical` | Vector of logits $\eta_t$ | $P(y_t=k)=\mathrm{softmax}(\eta_t)_k$ |

Do not return probabilities for discrete outcomes: GBM would apply the link
again. “Observation predictor” is the common term; a Gaussian mean is not in
general its canonical natural parameter. Categorical logits are invariant to a
common additive shift; redundant free offsets can create unidentifiable models.

### A learning model

```python
import jax
import jax.numpy as jnp
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit

def evolution(x, theta, u_t, y_t):
    alpha = jax.nn.sigmoid(theta[0])
    choice = y_t.astype(jnp.int32)
    return x.at[choice].add(alpha * (u_t["reward"] - x[choice]))

def observation(x, phi, u_t):
    inverse_temperature = jnp.exp(phi[0])
    return inverse_temperature * (x[1] - x[0])

model = StateModel(
    evolution=evolution, observation=observation, family="bernoulli",
    priors=Priors(
        evolution=GaussianPrior([0.0], [1.0], names=["alpha_raw"]),
        observation=GaussianPrior([1.0], [1.0], names=["log_beta"]),
    ),
    initial_state=[0.5, 0.5], state_names=["Q0", "Q1"],
)
# data: one dictionary per participant, with choice y and reward input u.
fit = individual_fit(data, model, config=Config(random_state=42))
```

This is a chosen-option Rescorla–Wagner update with a binary softmax choice rule.
The [README](../README.md) contains a complete runnable observation-only example.
Without `evolution`, omit its prior too; the default state `[0.0]` stays fixed.
If `observation` ignores that state, its value has no effect.

### Data and JAX rules

One subject: `individual_fit(y, model, u=u)`. Several subjects:
`individual_fit([{"y": y1, "u": u1}, {"y": y2, "u": u2}], model)`.
Subjects may have different trial counts. Gaussian outcomes have shape `(T,)`
or `(T, D)`; binary outcomes are 0/1; categorical outcomes are integer labels
`0, ..., K-1`. Several subjects must be separate dictionaries, not a stacked
outcome matrix. `u` is optional: `None`, a numeric trialwise array, or a dict
of numeric arrays with leading dimension `T` (scalar dict values are allowed).
No input field names are reserved. Missing/nonfinite outcomes are not supported.

Use `jax.numpy` inside model functions, `.at[...]` for array updates, and
JAX-compatible indexing/branching. Avoid NumPy operations on traced values,
Python `int`/`float` casts, in-place mutation and random draws. Functions must
be deterministic for fixed arguments. Preflight checks catch common errors,
but do not prove determinism or scientific validity for all inputs.

## 3. Specify priors and parameter coordinates

`GaussianPrior(mean, covariance)` accepts scalar variance, a diagonal variance
vector, or a full covariance matrix. **These are variances, not SDs.** A zero
marginal variance fixes that parameter at its mean; its cross-covariances must
also be zero. Fixed coordinates leave the optimization and evidence dimension.

The fitted order is `[theta, phi, log_process_sd, log_observation_sd]`.
Inspect `model.parameter_layout.names` rather than assuming positions.
The prior factorizes across these four blocks. Scientific priors may have
full covariance within the evolution or observation block; cross-block prior
correlations are not represented. The posterior Hessian can nevertheless
produce cross-block posterior correlations.

Constraints are usually expressed inside the model: `sigmoid(raw)` for a rate
in `(0,1)`, or `exp(raw)` for a positive quantity. A Gaussian prior on `raw`
induces a transformed prior on the scientific quantity. It is not a Gaussian
or uniform prior on that quantity. No extra Jacobian is needed when the prior
is defined directly on the fitted raw coordinate. A prior specified in a
different coordinate must be transformed with its Jacobian, as for Gamma noise.
MAP and the finite-order Laplace approximation depend on parameterization;
exact Bayesian probabilities transform consistently.

## 4. Specify state and observation noise

| Argument | Supported specification | Default |
|---|---|---|
| `initial_state_covariance` | Fixed scalar, diagonal vector or full matrix | Zero |
| `process_covariance` | Fixed covariance, callable `(theta, u_t)`, or `"diagonal"` with `process_noise_prior` | Zero |
| `observation_covariance` | Gaussian only: fixed covariance, callable `(phi, u_t)`, or estimated diagonal noise | Estimate noise |

State covariances may be positive semidefinite; Gaussian observation covariance
must be positive definite. Scalars mean variance times identity. Initial-state
covariance is not automatically estimated and does not accept a callable or
`"diagonal"` mode. Estimated process noise requires an explicit Gaussian prior
on log-SDs. The estimated forms are $Q=\mathrm{diag}(e^{2\rho_Q})$ and
$R=\mathrm{diag}(e^{2\rho_R})$. A callable permits parameter/input-dependent
covariance, but must remain valid over the parameter region being evaluated.
Binary/categorical outcomes do not use an additional observation covariance.

### Automatic Gaussian observation noise

Omitting observation covariance estimates noise under independent precisions

$$\lambda_j=\sigma_j^{-2}\sim\mathrm{Gamma}(a_j,b_j),\qquad a_j=b_j=1,$$

with **shape and rate**. The fitted coordinate is $\rho_j=\log\sigma_j$, whose
normalized log density includes the change-of-variables Jacobian:

$$\log p(\rho_j)=a_j\log b_j-\log\Gamma(a_j)+\log2
-2a_j\rho_j-b_j e^{-2\rho_j}.$$

Its negative log-prior curvature is $4b_j e^{-2\rho_j}$. Its moments are
$E[\rho_j]=(\log b_j-\psi(a_j))/2$ and
$\mathrm{Var}(\rho_j)=\psi_1(a_j)/4$. These moments scale optimizer
proposals; they do not replace the exact density with a Gaussian.
At $a=b=1$, they are about 0.2886 and 0.4112. The central 95% SD interval is
about `[0.52, 6.29]`; mean variance is infinite. Use quantiles to describe this prior.

```python
from gbmtoolbox import GammaPrecisionPrior
# Add one of these options to StateModel(...):
# observation_covariance=0.25                         # known variance
# observation_noise_prior=GammaPrecisionPrior(2, 0.5)  # shape, rate
# observation_noise_prior=GaussianPrior([0.0], [1.0])  # Gaussian on log-SD
```

For `D` outputs, set `observation_dim=D` or supply a matching vector noise
prior. A scalar prior is broadcast independently when the dimension is supplied.
The default estimates one precision per output, not their correlations or
trialwise changes. Matching VBA's Gamma(1,1) prior does not match its variational
algorithm or source-grouping of precisions.

`fit.output.observation_noise_sd` reports $e^{\hat\rho}$: the exponentiated
joint log-SD MAP, not the posterior mean or the SD-coordinate mode.
`fit.input.observation_noise_prior` records the actual resolved prior; its
mean/covariance alone cannot identify a non-Gaussian prior.

### Units and empirical priors

Gamma(1,1) is proper and fixed before data, but **not scale invariant**.
For $y'=cy$, $c\ne0$, consistently transform predictions, scientific priors and
noise rate $b'=c^2b$. Evidence then shifts by $-N\log|c|$ for $N$ scalar
observations, cancelling in Bayes factors on the same data. Resetting the rate
to 1 after changing units changes the prior model.

`observation_noise_prior_from_scale(y, prior_sample_size=nu0)` uses a robust
outcome scale $s_0$ to set $a=\nu_0/2$, $b=\nu_0s_0^2/2$.
Using the fitted outcomes to select it is empirical Bayes. Supplying
`prior_scale` from independent information avoids that data reuse. Constant
outcomes require an explicit scale. Do not tune priors to each model's residuals
and interpret the resulting evidence as comparison under a common measurement prior.

Changing the default noise prior affects Gaussian fits, evidence and downstream
comparisons. Discrete likelihoods and Gaussian models with explicit unchanged
noise specifications do not acquire this prior. Refit all candidates when
changing the prior; do not mix evidences from different defaults.

## 5. Understand MAP, uncertainty and evidence

GBM maximizes the normalized log joint over free coordinates:

$$\hat w=\arg\max_w\{\log p(y\mid w)+\log p(w)\}.$$

Multi-start L-BFGS-B uses four starts by default. Converged finite starts are
preferred, then the largest log joint among them. This is not proof of a global
optimum. The final observed Hessian is recomputed by automatic differentiation
(or `hessian_method="central_fd"`), independently of L-BFGS-B's approximation:

$$H=-\nabla_w^2\log p(y,w)\big|_{\hat w},\qquad
q(w)=\mathcal N(\hat w,H^{-1}),$$

$$\log Z_{\rm Laplace}=\log p(y\mid\hat w)+\log p(\hat w)
+\frac d2\log(2\pi)-\frac12\log\det H.$$

Here $d$ counts free coordinates, including estimated noise. Posterior SDs are
square roots of covariance diagonals; normal intervals assume a locally
Gaussian, single-mode posterior. The complete evidence expression balances
fit and prior/posterior concentration; the determinant term alone is not a
coordinate-independent complexity measure. Other posterior modes are not summed.

`laplace_valid` requires optimizer convergence and a positive-definite observed
Hessian. Otherwise covariance is `None` and evidence is `NaN`; MAP predictions
remain available. Hessian eigenvalues are not clipped. Ill conditioning or a
boundary optimum can set `laplace_fragile`. These flags do not certify model
adequacy, inner-filter convergence or approximation accuracy.

### Hard bounds

`Config(hard_bounds=...)` specifies one `(low, high)` pair or `None` per resolved
parameter; either endpoint may be `None`. These restrict the search. GBM does
not renormalize a truncated prior or compute a boundary-aware evidence integral.
If bounds represent scientific support, the returned evidence is not generally
the evidence of that normalized bounded model. A boundary mode also need not be
stationary for the unconstrained log joint. Use appropriate transformed priors
or a method that integrates the intended constrained distribution.

### Exact reference for Gamma noise

For fixed scalar predictions and deterministic states, let
$S=\sum_t(y_t-\mu_t)^2$, $A=a+T/2$, $B=b+S/2$. Then

$$\lambda\mid y\sim\mathrm{Gamma}(A,B),\quad
\hat\rho=\tfrac12\log(B/A),\quad H_{\rho\rho}=4A,$$

$$\log Z=-\tfrac T2\log(2\pi)+a\log b-\log\Gamma(a)
+\log\Gamma(A)-A\log B.$$

GBM still uses Laplace evidence, with known error

$$\log Z_{\rm Laplace}-\log Z
=(A-\tfrac12)\log A-A+\tfrac12\log(2\pi)-\log\Gamma(A).$$

This reference isolates the noise integral; it does not make evidence exact
when predictions or latent states are also uncertain.

## 6. Understand filtering

Filtering is part of likelihood evaluation whenever initial-state or process
covariance is nonzero, regardless of the requested uncertainty display:

$$
p(y_t\mid y_{1:t-1},w)=\int p(y_t\mid x_t,w)\,p(x_t\mid y_{1:t-1},w)\,dx_t
$$

The filter maintains a Gaussian approximation to the predicted state,
$\mathcal{N}(m_t,P_t)$. The fitted log likelihood sums its approximate predictive
log likelihoods. Outer Laplace evidence inherits any inner filtering error;
state marginalization is not a universal extra penalty.

### Gaussian outcomes: extended Kalman filter

With $J_t=\partial g/\partial x$ evaluated at $m_t$,

$$
S_t=J_tP_tJ_t^\top+R_t,\quad K_t=P_tJ_t^\top S_t^{-1}
$$

$$
m_t^{+}=m_t+K_t[y_t-g(m_t)],\quad
P_t^{+}=(I-K_tJ_t)P_t(I-K_tJ_t)^\top+K_tR_tK_t^\top
$$

The predictive log likelihood is $\log\mathcal{N}(y_t;g(m_t),S_t)$.
For transition Jacobian $F_t$ evaluated at $m_t^{+}$,

$$
m_{t+1}=f(m_t^{+},\theta,u_t,y_t),\qquad
P_{t+1}=F_tP_t^{+}F_t^\top+Q_t
$$

These are exact for linear-Gaussian models at default `filter_damping=1`, up
to floating-point error. Nonlinear evolution or observation gives an EKF
approximation; Gaussian observations alone do not imply exact integration.
Damping below 1 scales the gain in both mean and Joseph covariance updates;
this is not the ordinary EKF. The current Gaussian branch does not use
`filter_jitter`.

### Discrete outcomes: Fisher-scoring state update

Write $x=m+Ps$ to preserve a possibly singular state support. At the current
$x$, let $\ell(x)=\log p(y_t\mid x,w)$, $v=\nabla_x\ell(x)$ and
$C=J^\top WJ$, where $J$ differentiates the logits and $W$ is their Fisher
information: $p(1-p)$ for Bernoulli or $\mathrm{diag}(p)-pp^\top$ for
categorical observations. With damping $d_f$ and jitter $\epsilon$, the code uses

$$
s\leftarrow s+d_f[(1+\epsilon)I+CP]^{-1}(v-s)
$$

After at most `filter_max_iter` iterations, the approximate covariance and
predictive log likelihood are

$$
P^{+}=\mathrm{sym}([(1+\epsilon)I+PC]^{-1}P)
$$

$$
\log\widetilde{p}(y_t)=\ell(m+Ps)-\frac{1}{2}s^\top Ps
-\frac{1}{2}\log\det[(1+\epsilon)I+PC]
$$

At convergence and zero jitter, this is the usual local Laplace calculation
when the logits are affine in the state. For nonlinear logits, Fisher curvature
omits second derivatives of the observation map and is generally not the exact
observed state-posterior Hessian. Finite iterations add another approximation;
there is no guarantee the update reaches the mode. These approximate masses
are not guaranteed to normalize exactly over all possible outcomes.

`filter_jitter` regularizes discrete solves; it does not add variance to null
state directions. Avoiding covariance eigenvectors makes derivatives well
behaved at repeated eigenvalues, but changing covariance rank remains delicate.

### When to use filtering and what to inspect

Use filtering when state/process uncertainty is part of the scientific model.
Local Gaussian approximations are most plausible when distributions remain
concentrated and approximately unimodal. Strong nonlinearities, broad or
multimodal states, and weakly observed dynamics need independent checks such
as numerical integration or simulation-based reference inference.

Inspect the inner update separately:

```python
run = fit.model.evaluate(fit.output.parameters[0], fit.data[0], config=fit.profile.config)
print(run.get("update_converged"))
print(run.get("update_iterations"))
```

The optimizer currently uses the computed likelihood even if an inner discrete
update has not converged. Increasing `filter_max_iter` and checking stability
can expose truncation error; outer `laplace_valid=True` does not rule it out.
For discrete stochastic models, `prediction` is sigmoid/softmax at $m_t$,
not the probability integrated over the state distribution. It need not equal
the exponentiated approximate observed-outcome log likelihood above.

## 7. Read latent uncertainty and predictive simulations

| `Config(latent_uncertainty=...)` | What is reported |
|---|---|
| `"none"` | State trajectory without uncertainty intervals |
| `"propagated"` | Spread of trajectories/filtered means across static Laplace parameter draws |
| `"filtered"` | Filtered state covariance conditional on the fitted static parameters |

Propagated draws rerun the model with the **observed** outcomes. For stochastic
models they rerun the filter and retain its state means; they do not draw state
paths. They omit the conditional variance contribution in

$$\mathrm{Var}(x_t\mid y)=
E_w[\mathrm{Var}(x_t\mid y,w)]+\mathrm{Var}_w[E(x_t\mid y,w)].$$

This identity explains the two sources of uncertainty. GBM propagates filtered
means rather than smoothed means, so it does not compute the full-data second
term exactly either. It is not a full smoothing posterior over states and parameters. Filtered
output conditions on the MAP and uses observations up to that trial, not future
observations. Neither option alone supplies total state uncertainty in a
stochastic model. `"filtered"` is rejected for a deterministic state model.
Deterministic trajectories are recorded before evolution; stochastic `states`
are filtered means, while `predicted_state` gives their pre-observation values.

`prior_predictive(model, template_data, ...)` samples parameter priors (including
exact Gamma precision draws), state noise and outcomes. `posterior_predictive`
samples the joint Gaussian Laplace parameter approximation and generates new
state paths/outcomes. Generated outcomes feed subsequent evolution; trial
inputs are copied from the template. If rewards or stimuli depend on simulated
choices, supply a simulator for that task/environment rather than treating a
recorded adaptive input sequence as a freely generated experiment.

These simulations check model adequacy; they do not establish that parameters
or competing models are identifiable. Posterior-predictive draws require valid
Laplace covariance and do not incorporate uncertainty over learned group priors.

## 8. Check a fit

Start with `fit.math.diagnostics[subject]`. Inspect convergence across starts,
posterior curvature, parameter plausibility, and predictive discrepancies.
A finite evidence alone is insufficient.

### Local prediction-sensitivity diagnostic

The public function remains `numerical_local_identifiability(fit, subject=0)`.
It computes the singular values and rank of the prediction Jacobian at the MAP,
with one redundant categorical probability removed. A null direction leaves
reported predictions unchanged to first order. This is neither a proof of global
identifiability nor a test of the entire likelihood: a variance parameter can
have zero mean-prediction sensitivity while being identifiable from dispersion.
Use noise correlations and likelihood curvature alongside this diagnostic.

### Likelihood curvature in prior units

`prior_preconditioned_information` computes

$$H_{\rm lik}=H-H_{\rm prior}(\hat w),\qquad
M=\Sigma_0^{1/2}H_{\rm lik}\Sigma_0^{1/2}.$$

For Gaussian priors, an eigenvalue above 1 means likelihood curvature exceeds
prior curvature along the corresponding scaled direction. For Gamma noise,
the code subtracts its actual local curvature but scales by global log-SD
prior moments. Thus the same threshold is a variance-scaled descriptive index,
not an exact comparison with local Gamma prior curvature. Negative values
indicate negative observed likelihood curvature; they do not by themselves
invalidate a positive-definite posterior Hessian.

### Observation-noise correction: a residual-energy diagnostic

`observation_noise_correction` asks: **how much does uncertainty in fitted
predictions add to squared residuals?** It does not decide whether the fitted
observation SD is correct. Let $r_t=y_t-g_t(\hat w)$, and linearize the prediction
under $q(w)=\mathcal N(\hat w,\Sigma)$:

$$E_q[(y_t-g_t(w))^2]\approx r_t^2+J_t\Sigma J_t^\top.$$

The cross term vanishes because $E_q[w-\hat w]=0$. “Residual energy” means the
sum of squared prediction errors, not a separate physical quantity. For each
output dimension the diagnostic reports

$$s_{\rm residual}=\sqrt{SSE/T},\qquad
s_{\rm corrected}=\sqrt{(SSE+U)/T},\qquad
U=\sum_tJ_t\Sigma J_t^\top.$$

`inflation_factor` compares these two residual scales. `reported_sd` separately
records the fitted noise summary. A large factor indicates prediction uncertainty
contributes substantially; a factor near 1 does not validate the prior, model,
posterior approximation or reported noise. There is no universal 1.1 cutoff.
This is a first-order prediction approximation, not an exact nonlinear expected
residual; prediction curvature can contribute additional terms.

**It is not generally unbiased.** In full-rank Gaussian linear regression with
$p$ mean coefficients and effectively flat priors, inserting the ML covariance
$\Sigma=\hat\sigma^2(X^\top X)^{-1}$ gives $U=p\hat\sigma^2$, and one step yields

$$s_{\rm corrected}^2=\frac{SSE}{T}(1+p/T),\qquad
E[s_{\rm corrected}^2]=\sigma^2[1-(p/T)^2].$$

Solving the different self-consistency equation
$s^2=(SSE+ps^2)/T$ gives $SSE/(T-p)$, the unbiased **variance** estimator under
those assumptions. GBM does not perform that iteration. Taking the square root
also introduces bias: with $\nu=T-p$,

$$E[\sqrt{SSE/T}]=\sigma\sqrt{2/T}\,
\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)},$$

not exactly $\sigma\sqrt{(T-p)/T}$. Proper priors, nonlinear models, and joint
noise estimation alter these linear-reference results. In particular, the
Gamma-prior MAP noise is not generally the ML residual variance.

The diagnostic changes neither the fitted posterior nor model evidence. It is
available for deterministic Gaussian models with free parameters and valid
Laplace covariance; it can also be computed with fixed observation covariance.
It summarizes each output's residual scale, not a full covariance correction or
a trial-specific heteroscedastic noise estimate. For stochastic states it returns
`valid=False`, because predicted-state residuals also contain state uncertainty.

## 9. Compare models, families, conditions and groups

All candidates must describe the same observed data and observation measure.
Do not compare a continuous density with a mass for differently encoded data.
Use defensible normalized priors and inspect every subject's fit first.

### Basic random-effects BMS

Let $L_{sk}$ be subject $s$'s log evidence for model $k$. The population model is

$$r\sim\mathrm{Dirichlet}(\alpha_0),\quad
z_s\mid r\sim\mathrm{Categorical}(r),\quad
p(y_s\mid z_s=k)=e^{L_{sk}}.$$

Mean-field inference alternates responsibilities and counts:

$$\gamma_{sk}\propto\exp[L_{sk}+\psi(\alpha_k)-\psi(\sum_j\alpha_j)],\qquad
\alpha_k=\alpha_{0k}+\sum_s\gamma_{sk}.$$

Normalize responsibilities over models. Exactly constant evidence rows are
marginalized before count updates: they contribute an evidence offset but no
frequency information. Weak nonconstant rows can still suffer mean-field bias.

| Output | Interpretation |
|---|---|
| `model_frequency` | Posterior expected population frequency |
| `responsibilities` | Subject-specific model assignments, probabilistically |
| `exceedance_prob` (EP) | Probability a frequency exceeds every other frequency |
| `bor` | Posterior probability of a fixed equal-frequency null under equal hypothesis prior odds |
| `protected_exceedance_prob` (PXP) | $(1-\mathrm{BOR})\mathrm{EP}+\mathrm{BOR}/K$ |

The alternative's `log_evidence` is a variational lower bound for model-only
BMS. The null is $F_0=\sum_s\log[(1/K)\sum_ke^{L_{sk}}]$ and
$\mathrm{BOR}=\mathrm{sigmoid}(F_0-F_1)$. Approximation errors in bounds
need not cancel in hypothesis odds. EP is not the probability that a model is
universally true or that every subject uses it.

### Current defaults and family choices

| Setting | Default in this version |
|---|---|
| Model prior without families | $\alpha_{0k}=1/K$ (one total count) |
| With $F$ families of sizes $K_f$ | $\alpha_{0k}=1/(FK_f)$ for $k\in f$ |
| `family_method` | `"aggregate"` |
| `group_level` | `"models"`, even when families are supplied |

Explicit `alpha0` overrides prior counts. Use `np.ones(K)` for one count per
model. Families must be nonempty, disjoint, zero-based and cover all columns.

**Aggregate:** sum the model-level variational Dirichlet counts and assignments
within families. This is exact aggregation of an approximate posterior. It
retains a population distribution over the individual models. Weak evidence,
unequal family sizes and small prior counts can produce large mean-field bias.

**Marginal:** integrate a uniformly chosen within-family model separately for
each subject, then infer family frequencies:

$$L_{sf}=\mathrm{logsumexp}_{k\in f}L_{sk}-\log K_f,\qquad
\alpha_{0f}=\sum_{k\in f}\alpha_{0k}.$$

This does not estimate population frequencies within each family. Select it
explicitly when that is the intended model:

```python
from gbmtoolbox import bms
result = bms(L, families=[[0, 1], [2, 3]], family_method="marginal")
```

For two marginal families, numerical quadrature integrates

$$p(r\mid y)\propto r^{a-1}(1-r)^{b-1}
\prod_s[re^{L_{s0}}+(1-r)e^{L_{s1}}].$$

It returns integrated means, variances, assignments, evidence and EP, up to
numerical error. This posterior is generally not Beta, so
`posterior_parameters=None`; `expected_counts` are not exact Dirichlet parameters.
More than two marginal families and all model-only BMS reports use mean-field
inference. Duplicating every model within a family leaves its marginal evidence
unchanged; aggregation need not have this invariance.

The family null uses probability $1/(FK_f)$ for each model in family $f$.
Family BOR compares it with the full-model bound for aggregate inference, or
the family evidence/bound for marginal inference. Family PXP uses this family
BOR. `families.within[f]` is a separate analysis restricted to that family,
with a uniform prior totaling one; it is not the full posterior renormalized.
A singleton has frequency and EP/PXP one; its vacuous BOR is one.

### Paired and factorial conditions

`bms_conditions(L)` accepts `(subjects, models, conditions)`; subject identities
and row order must match. A model tuple $(m_1,\ldots,m_C)$ has evidence
$L_{s,t}=\sum_cL_{s,m_c,c}$. This assumes separately integrated condition-specific
parameters with independent priors and appropriate conditional independence.
Shared parameters require joint evidence; overlapping observations must not be
counted twice.

Tuples are `equal` if all models are identical, or all belong to the same supplied
family. The equal/not-equal tuple families receive equal prior mass, total count
one; `alpha0` controls only the per-condition analyses. `family_method` also
controls tuple-family inference; use `"marginal"` explicitly for integration.
`xp` and `pxp` concern the prevalence of equal versus unequal tuples, not a
probability that all subjects share one model. `best` uses the model-tuple
variational frequencies even when tuple-family reports use marginal integration.

`factors=np.array([[0,1],[2,3]])` defines a complete factorial condition layout.
Each non-singleton axis tests equality along that axis at all combinations of
other levels; `factor_axes` records the axes. Enumeration requires $K^C$ tuples:
`max_tuples=100000` limits allocation, with a warning above 10,000. One condition
returns ordinary BMS; one family makes equality vacuous.

### Independent groups

`bms_groups([L_a, L_b])` allows unequal group sizes, with matching model columns.
It compares one shared population-frequency profile ($H_=$) against separate
profiles ($H_{\ne}$):

$$F_=F(\mathrm{stack}_g L_g),\quad F_{\ne}=\sum_gF(L_g),\quad
\log BF_{\ne,=}=F_{\ne}-F_=,$$
$$P(H_=\mid y)=\mathrm{sigmoid}[F_=-F_{\ne}+\log(\pi_=/(1-\pi_=))].$$

`prior_equal=0.5` sets $\pi_=$; each profile has the same specified Dirichlet
prior. Two comparison categories use integrated frequency evidence; larger
spaces use variational bounds. `evidence_method` identifies the calculation.
`comparison_pooled` and `comparison_per_group` remain variational BMS summaries
even when the top-level group odds use binary quadrature.

The default compares **model frequencies**. For family frequencies, use

```python
from gbmtoolbox import bms_groups
out = bms_groups([L_a, L_b], families=[[0, 1], [2, 3]], group_level="families")
print(out.probability_equal, out.probability_different)
```

This uniformly marginalizes within-family models and uses summed family prior
counts. `group_level="auto"` selects families when supplied, otherwise models.
These hypothesis probabilities are not EPs. Groups can have different model
frequencies but identical family frequencies. `family_method` controls attached
family reports, whereas `group_level` controls the group hypothesis itself.

`bms` dispatches a 2D array to ordinary BMS, a 3D array to paired conditions,
and a list of 2D matrices to independent groups. Dirichlet EPs use Beta tails
for two candidates and seeded, memory-bounded Monte Carlo otherwise. At the
default 100,000 draws, each sampled EP has Monte Carlo SE at most 0.0016.
The VBA-inspired constructions differ in family nulls, optional marginal
families and condition protection; they are not a line-by-line MATLAB port.

## 10. Refit subjects hierarchically

`hierarchical_fit` alternates subject fitting and learned Gaussian group priors.
It is **empirical-Bayes/Laplace refitting**, inspired by CBM/HBI, not the original
variational HBI algorithm. Use the current public names:

```python
from gbmtoolbox import Config, HierarchicalFitConfig, hierarchical_fit
result = hierarchical_fit(
    data, [model_a, model_b],
    fit_config=Config(random_state=42, verbose=False),
    hbi_config=HierarchicalFitConfig(maxiter=30, tol=1e-3, verbose=False),
)
```

The keyword `hbi_config` is retained by the current API; it does not indicate
full HBI inference. Old imports `hbi_main` and `HBIConfig` are not exported.

For subject posterior block means $m_{sk}$, covariances $S_{sk}$ and
responsibilities $\gamma_{sk}$, write $N_k=\sum_s\gamma_{sk}$. The proposed group
update is

$$\mu_k=\frac1{N_k}\sum_s\gamma_{sk}m_{sk},\qquad
\Sigma_k=\frac1{N_k}\sum_s\gamma_{sk}
[S_{sk}+(m_{sk}-\mu_k)(m_{sk}-\mu_k)^\top].$$

Responsibilities use current subject Laplace evidence and expected log model
frequencies, followed by $\alpha_k=1+\sum_s\gamma_{sk}$. This hierarchical
routine uses one count **per model**, independently of the standalone BMS default.
One responsibility update occurs per outer sweep.

Group updates are separate for evolution and observation blocks. Covariances
are full within each block, not across blocks; noise priors remain fixed.
Fixed scientific coordinates remain fixed. Near-zero model weight retains the
original block prior. Free group covariance eigenvalues are floored at
$\max(10^{-8},10^{-8}\lambda_{\max})$; this regularization is distinct from the
unrepaired subject posterior Hessian.

Stopping checks maximum absolute changes in group means, covariance entries
and model frequencies. One tolerance is applied in these different units, so
parameter scaling matters. Returned `group_priors` are those used for returned
subject fits; the last proposed update is not applied without a refit.
Inspect `converged` and `convergence_history`; reaching `maxiter` is not convergence.

There is no group hyperprior, integrated uncertainty over group means/precisions,
or monitored hierarchical free-energy bound. Learned-prior subject evidences
are conditional empirical-Bayes quantities, not fully integrated hierarchical
model evidence. Passing them to `bms` gives a conditional comparison and does
not recover the original HBI method. Group-frequency responsibilities also
inherit the approximate subject fits and the responsibility-update assumptions.

## 11. Control speed and reproducibility

Deterministic observation-only models use vectorized likelihoods; evolving
states use sequential scans. Scalar/diagonal Gaussian covariances use elementwise
formulas. Objective and evaluator compilation can be reused across prior changes.
Callables and captured settings must remain immutable; new shapes, dtypes,
input structures or callables can require compilation.

`Config(parameter_scaling=True)` changes search coordinates to $w=\mu+Bz$.
Without bounds, $B$ is a prior-covariance Cholesky factor; with bounds it uses
marginal SDs to preserve box constraints. The log-joint objective remains in
original coordinates, without an added coordinate Jacobian. Returned curvature
uses $H_w=B^{-T}H_zB^{-1}$ and evidence remains in the original coordinates.
Scaling is off by default and can change convergence paths/local optima.

`Config(n_jobs=4)` fits subjects in separate processes; `-1` uses available CPUs.
Worker startup and compilation can outweigh gains for short fits. Do not assume
a benchmark speedup transfers to a different model or hardware.

`Config(random_state=42)` seeds individual optimization and parameter propagation
through separate streams by subject index. Reordering subjects changes their
assigned streams. Hierarchical fitting inherits that seed unless an explicit
`HierarchicalFitConfig` is supplied; each model retains its seed across sweeps.
BMS and predictive functions have separate `random_state` arguments.
`None` requests unseeded draws. Seeded results need not be bitwise identical
across JAX versions or hardware.

A seed does not fix runtime: compilation, caches, process startup and load vary.
Unless already configured, a persistent JAX cache is set at
`~/.cache/gbmtoolbox/jax`; disable it with
`jax.config.update("jax_enable_compilation_cache", False)`.
Inspect `fit.profile.timings` and `fit.math.diagnostics[s].timings`,
`n_evaluations` and start records. Separate cold and warm fits; nested timing
stages must not be added together. Custom JAX timers must synchronize results.

## 12. Report an analysis and find references

Report the toolbox commit and dependencies; model equations and outcome coding;
priors and fitted coordinates; fixed parameters, noise structure and units;
optimizer/filter settings and diagnostics; uncertainty approximation; and BMS
prior counts, family method and group comparison level. For hierarchical fitting,
report convergence and identify the empirical-Bayes approximation explicitly.

The [validation guide](VALIDATION.md) distinguishes exact reference calculations,
approximation checks and empirical recovery examples. Passing selected checks
does not establish accuracy for an arbitrary nonlinear model.

- Piray et al. (2019), [Hierarchical Bayesian inference for concurrent model
  fitting and comparison for group studies](https://doi.org/10.1371/journal.pcbi.1007043).
  Methodological lineage; GBM's empirical-Bayes refitting differs as described above.
- [VBA group-comparison documentation](https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/)
  and [source](https://github.com/MBB-team/VBA-toolbox): reference statistical constructions,
  with the GBM differences stated in section 9.
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html): compilation and timing.
