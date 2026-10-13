# Validation

How the toolbox is checked, and what the checks currently measure.

Validation has two layers. The automated suite in `tests/` is the merge gate:
it is fast, runs in CI, and must be green. The reference scripts in
`gbmtoolbox/dev/` are the scientific layer: each one poses a mathematical
question, computes an independent reference answer, and reports the error
against it. The scripts print numbers rather than only asserting, because the
size of the error is itself the result.

Where an analytic answer exists, it is the reference. Where it does not, the
reference is high-accuracy numerical integration or an exactly solvable special
case. A code-to-code comparison is never treated as validation.

## Current status

| | |
|---|---|
| Python | 3.11.14 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| JAX / jaxlib | 0.10.2 |
| Platform | macOS arm64 |
| Date | 2026-10-13 |

**Automated suite: 174 passed, 0 failed. Reference scripts: 15 of 15 finish;
the 14 with a reference answer pass, and script 13 reports timings only.**
All 11 tutorials in `gbmtoolbox/examples/` also run to completion. There is no
CI configuration in the repository; the results above are from a local run.

## Running all checks

```bash
# merge gate; run the HBI/predictive file on its own to bound peak memory
python -m pytest -q --ignore=tests/test_predictive_bms_hbi.py
python -m pytest -q tests/test_predictive_bms_hbi.py
# scientific layer
for f in gbmtoolbox/dev/*.py; do python "$f"; done
```

Each section below names its script, the question it answers, the reference,
the pass criterion and the measured result.

---

## 1. Family likelihoods against closed-form formulas

`01_likelihood_reference_cases.py`

**Question.** Does the state model produce the same per-trial log likelihood as
the elementary Gaussian, Bernoulli and categorical formulas?

**Reference.** Written out directly. For a Bernoulli outcome with success
probability $p$, $\log p(y) = y\log p + (1-y)\log(1-p)$; for a categorical
outcome, $\log p_k$; for a Gaussian outcome,
$-\tfrac{1}{2}\left[\log(2\pi\sigma^2) + (y-\mu)^2/\sigma^2\right]$.

**Criterion.** Absolute error above `1e-12` fails.

**Result.** Maximum absolute error `0.0` for all three families — the
implementation reproduces the formulas exactly in these deterministic cases.

## 2. Observed-Hessian conditioning without clipping

`02_hessian_failure_cases.py`

**Question.** When a model is deliberately near-unidentifiable, is the
resulting ill-conditioning reported rather than silently repaired?

**Reference.** A posterior chosen so the answer is known exactly,

$$L(x,y) = \tfrac{1}{2}x^2 + \tfrac{1}{2}\varepsilon y^2 + \tfrac{1}{4}10^{-26}y^4,$$

whose Hessian at the MAP is $\mathrm{diag}(1, \varepsilon)$, so
$\mathrm{cond}(H) = 1/\varepsilon$ exactly.

**Criterion.** The recovered $\lambda_{\min}$ must track $\varepsilon$, and no
eigenvalue may be clipped to force positive definiteness.

**Result.** Recovered $\lambda_{\min}$ matches $\varepsilon$ across
`1e-10` to `1e-14`, with condition numbers `1e10` to `1e14` reported as such.
No clipping was applied. A valid MAP and a valid Laplace approximation remain
separate outcomes: the toolbox reports an invalid Laplace rather than
manufacturing a usable covariance.

## 3. Linear-Gaussian filtering against the exact Kalman recursion

`03_linear_gaussian_filter.py`

**Question.** On a linear-Gaussian model, where the Kalman filter is exact,
does the general filter reduce to it?

**Reference.** A hand-written scalar Kalman recursion, trial by trial, over
filtered means, variances and predictive log likelihood.

**Criterion.** Absolute error above `1e-7` fails.

**Result.**

| quantity | max error |
|---|---|
| filtered mean | 6.66e-11 |
| filtered variance | 7.75e-11 |
| predictive log likelihood | 2.33e-10 |

The residual is not approximation error in the filter. It is set by
`Config.filter_jitter`, the diagonal regularizer added inside the
symmetric-positive-definite solves, and scales linearly with it:

| `filter_jitter` | max error |
|---|---|
| `1e-9` (default) | 2.3e-10 |
| `1e-12` | 2.3e-13 |
| `1e-14` | 2.3e-15 |
| `0` | 1.1e-16 (machine precision) |

With jitter disabled the filter reproduces the exact Kalman recursion to
machine precision. The default trades roughly three orders of accuracy for
numerical robustness on poorly conditioned covariances.

## 4. Bernoulli filtering against quadrature

`04_bernoulli_filter_quadrature.py`

**Question.** The discrete filter is approximate. How large is the
approximation error in a case where the exact answer can be computed?

**Reference.** The one-step predictive probability is a one-dimensional
integral against the Gaussian state density,

$$P(y_t = 1) = \int \sigma(x)\, \mathcal{N}(x \mid \mu, \sigma^2_x)\, dx,$$

evaluated by adaptive quadrature to `epsabs=1e-12`. Here $\mu = 0.4$,
$\sigma^2_x = 0.5$.

**Criterion.** Absolute error above `0.04` fails.

**Result.** Exact `0.58896755`, filter `0.58773624`, **absolute error
`1.231e-03`**.

## 5. Categorical filtering against quadrature

`05_categorical_filter_quadrature.py`

**Question.** The same question for a three-category outcome.

**Reference.** Quadrature against the Gaussian state density with logits
$(x, 0, -x)$, $\mu = 0.2$, $\sigma^2_x = 0.7$.

**Criterion.** Absolute error above `0.06` fails.

**Result.** Exact `0.30911977`, filter `0.30498378`, **absolute error
`4.136e-03`**.

Checks 4 and 5 bound the approximation in specific, documented cases. They do
**not** establish that the discrete filter is exact, or uniformly accurate, for
arbitrary nonlinear or non-Gaussian models. It is a local Laplace-Gaussian
approximation with Fisher scoring and is labelled as approximate throughout.

## 6. Propagated uncertainty against analytic propagation

`06_propagated_uncertainty.py`

**Question.** Does sampling static parameters from the Laplace posterior and
rerunning the model reproduce the analytic variance in a linear case where that
variance is known in closed form?

**Reference.** Exact linear propagation of a Gaussian, where the transformed
variance is available analytically.

**Criterion.** Monte-Carlo relative error consistent with the sample size.

**Result.** Analytic `[0.20, 0.80, 0.45, 3.20]` against sampled
`[0.2011, 0.8044, 0.4525, 3.2177]` — **relative error 0.552%** on every
component, as expected for the sample size used.

## 7. Numerical local identifiability on a known ridge

`07_information_identifiability.py`

**Question.** In a model built to be rank-deficient, is the redundancy
detected and the null direction recovered?

**Reference.** A Bernoulli model whose logit is $\phi_0 + \phi_1$. Only the sum
is observable, so the information matrix has rank 1 and the null direction is
known to be $\propto (1, -1)$.

**Criterion.** Reported rank 1, nullity 1.

**Result.** Singular values `[1.0, 6.64e-17]`, rank/nullity `1 / 1`, weak
direction `[0.7071, -0.7071]` — the analytic null direction, recovered to
numerical precision.

This diagnostic is a numerical rank and sensitivity check at the fitted point.
It is not a proof of global structural identifiability.

## 8. Automatic differentiation against finite differences

`08_ad_derivatives.py`

**Question.** The default Hessian is computed by automatic differentiation. Does
it agree with the central finite-difference Hessian it replaced?

**Reference.** The same fit run with `hessian_method="central_fd"`.

**Result.** MAP difference `0.0`; Hessian maximum absolute difference
`1.53e-04`, consistent with finite-difference truncation error rather than a
disagreement between the two methods.

## 9. Noise-parameter recovery

`09_noise_recovery.py`

**Question.** When process and observation noise are estimated rather than
fixed, are they recovered, and does the posterior reflect their joint curvature?

**Reference.** Data simulated from known $Q = 0.08$ and $R = 0.30$.

**Result.** Recovered $Q = 0.0926$, $R = 0.3123$. The posterior correlation
matrix carries non-zero joint $Q/R$ curvature, confirming the noise
parameters enter the full posterior rather than being profiled out.

## 10. Default observation-noise prior against the analytic posterior mode

`10_default_observation_noise.py`

**Question.** When `observation_covariance` is omitted, a Gaussian model
estimates $R$ under the default prior: VBA's $\mathrm{Ga}(1,1)$ on the
observation precision $\lambda=\sigma^{-2}$, evaluated exactly in
$\rho=\log\sigma$. Does the fitted value match what that model implies
analytically?

**Reference.** For a constant-mean Gaussian with the mean fixed and
$S=\sum_t (y_t-\mu)^2$, the log posterior in $\rho$ with a
$\mathrm{Ga}(a,b)$ precision prior is

$$L(\rho) = -(T+2a)\rho - \left(\tfrac{S}{2}+b\right)e^{-2\rho} + \text{const},$$

so the mode is $\hat\sigma^2=(S+2b)/(T+2a)$, i.e.
$\hat\sigma=\sqrt{(1+S/2)/(1+T/2)}$ at $a=b=1$. This is the exponentiated
log-SD mode, not the mode or mean of $\sigma$ in its own coordinate.

**Criterion.** Relative error above `1e-3` against the analytic mode fails.

**Result.** Worst relative error `3.3e-08` across four cases spanning
$\sigma \in [0.3, 2.5]$ and $T \in [200, 800]$.

---

## 11. Observation-noise correction for parameter uncertainty

`11_observation_noise_correction.py`

**Question.** The estimated observation SD is driven by the residuals at the
fitted parameters, which treats those parameters as known exactly. Does that
reproduce the biased maximum-likelihood variance, and does correcting the
residual energy with the prediction uncertainty
$\sum_t J_t \Sigma J_t^\top$ remove the bias?

**Reference.** For a linear model with $p$ parameters the maximum-likelihood
estimator $\sqrt{SSE/T}$ is biased low by exactly $\sqrt{(T-p)/T}$, and
$\sum_t J_t \Sigma J_t^\top = \sigma^2 p$, so the corrected estimate is the
unbiased $SSE/(T-p)$. Both are known in closed form.

**Criterion.** The tolerance is expressed in Monte-Carlo standard errors, not
in absolute SD units. The estimator has standard error
$\sigma / \sqrt{2(T-p)N}$ over $N$ replicates — about `0.02` at $T = 25$,
$p = 5$, $N = 60$ — so any absolute tolerance tight enough to be meaningful at
$T = 200$ would sit below the noise floor at $T = 25$ and fail at random.
A deviation above `4` standard errors fails.

**Result.** Across $T \in \{25, 50, 100, 200\}$ and $p \in \{3, 5\}$, the
uncorrected estimate reproduces the analytic maximum-likelihood bias to within
`1.77` standard errors, and the corrected estimate matches the true SD to
within `2.37`. The correction is applied once at the MAP rather than iterated
to self-consistency, so a small downward bias survives at the smallest trial
counts and shrinks as $T$ grows.

---

## 12. Exact Gamma noise prior: density, evidence and Bayes factors

`12_gamma_precision_evidence.py`

**Question.** With the default `Gamma(1, 1)` precision prior evaluated in
`log_observation_sd`, are the MAP, the observed Hessian and the Laplace
evidence correct, and do Bayes factors and BMS inputs inherit that accuracy?

**Reference.** With fixed predictions the Gamma precision integrates
analytically: the exact log-SD posterior mode, its curvature, and the exact
marginal likelihood are closed form. Laplace evidence is not exact; it is
compared with its known Stirling-series error, not with the exact evidence.

**Criterion.** Assertions on the mode, Hessian and predicted Laplace error, on
a Bayes factor between two such models, and on BMS invariance to a common
evidence shift.

**Result.** Over $T \in \{1, 5, 25, 100, 1000\}$ the largest errors are
`7.6e-08` (mode), `1.5e-07` (relative Hessian) and `7.6e-08` (Laplace minus
its predicted error). The log Bayes factor error is `5.3e-09`; the BMS
common-shift error is `6.4e-22`.

---

## 13. Fit timings

`13_fit_performance.py`

**Question.** How long do cold (first, compiling) and warm (repeated) fits take
for an observation-only model with each outcome family?

**Reference.** None: this script reports timings and asserts nothing about
them. It uses unscaled optimizer coordinates, the library default; `--scaled`
times `Config(parameter_scaling=True)`. Disk compilation caching is disabled,
so cold times include compilation.

**Result.** A before/after comparison was made on 2026-10-13 against commit
`c28a46f` on Linux x86_64 (Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3,
JAX 0.4.38, CPU), with 2,000 trials, two observation coefficients, one start,
estimated Gaussian noise under the default prior, and parameter scaling
enabled. Warm results are medians of five further fits on the same model.

| Outcome | Before, first fit (s) | After, first fit (s) | Before, warm (s) | After, warm (s) | Warm speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 1.466 | 0.648 | 0.428 | 0.00701 | 61.0x |
| Bernoulli | 0.567 | 0.409 | 0.179 | 0.00512 | 35.0x |
| Categorical | 0.772 | 0.415 | 0.227 | 0.00523 | 43.4x |

Maximum fitted-parameter differences were `4.90e-7` (Gaussian) and below
`2e-16` (discrete). Absolute log-evidence differences were `1.47e-6` and zero.
These examples do not establish a speedup for nonlinear dynamics, stochastic
filtering, other hardware, or a particular user model. To reproduce:

```bash
python gbmtoolbox/dev/13_fit_performance.py --trials 2000 --repeats 5
python gbmtoolbox/dev/13_fit_performance.py --trials 2000 --repeats 5 --scaled
```

The same update is covered by `tests/test_performance_paths.py`: vectorized
observation-only likelihoods, gradients and Hessians against the sequential
formulas (tolerance `2e-11`); every Gaussian covariance path against the dense
normalized likelihood; scaled fits against the analytic linear-Gaussian
posterior (Hessian to `1e-10`, covariance and evidence to `1e-12`/`1e-8`);
and compilation reuse after prior values change, compared with an independently
compiled model, so stale prior values in cached executables are detected.

---

## 14. Family, condition and group model selection

`14_bms_families_conditions_groups.py`

**Question.** Are the random-effects evidences, group hypothesis probabilities,
exceedance probabilities, family nulls and condition-tuple evidences correct?

**Reference.** Independent mathematical cases:

- **RFX evidence.** With effectively known model labels, counts $n$ and
  Dirichlet prior $a_0$, the evidence is $\log B(a_0+n)-\log B(a_0)$. Pooled
  and separate group bounds must reproduce it.
- **Group hypothesis probability.** The pooled Dirichlet integral against the
  product of independent group integrals, under the stated prior odds.
- **Binary exceedance.** The Beta survival probability at one half.
- **Family null.** A directly evaluated, normalized finite mixture, including
  unequal family sizes.
- **Condition tuples.** Direct products of per-condition likelihoods.

**Criterion.** Every absolute error below `1e-10`.

**Result.**

| Reference | Absolute error |
| --- | ---: |
| Shared-frequency evidence | `7.8e-14` |
| Separate-frequency evidence | `1.5e-13` |
| Group hypothesis probability | `1.7e-15` |
| Binary exceedance | `0.0` |
| Family-null mixture evidence | `2.2e-16` |
| Known condition-tuple evidence | `2.7e-15` |

These establish correctness in the reference cases, not exactness of general
mean-field RFX evidence: general group probabilities use differences of
variational bounds, whose approximation errors need not cancel. VBA-style
constructions were checked against the VBA MATLAB sources; MATLAB itself was
not run. `tests/test_bms_extended.py` adds sampled checks of family
aggregation, factorial condition axes, offset invariance and input validation.

---

## 15. Weak-evidence model selection and multistate filter derivatives

`15_reliability_reproducibility.py`

**Question.** When evidence barely distinguishes two candidates, do the
two-family frequency posterior and the two-group hypothesis probability stay
exact rather than drifting under the mean-field approximation? And are the
gradient and Hessian of a two-state filter likelihood with an isotropic
covariance (repeated eigenvalues) finite and correct?

**Reference.** Direct Beta-weighted quadrature of the frequency likelihood for
the BMS cases; an exactly uninformative pair of groups must return the prior
probability. Central finite differences for the filter derivatives.

**Criterion.** BMS errors below `1e-10`; finite-difference agreement within
`1e-6` (gradient) and `2e-5` (Hessian).

**Result.** All BMS errors are at most `4.6e-16`; the uninformative group
probability is exact. Filter derivative errors are `2.7e-09` (gradient) and
`2.1e-08` (Hessian).

Scope: exact integration is restricted to two marginal families and
two-category group odds. Model-only BMS reports, aggregate families and larger
group spaces keep their variational approximations. The discrete filter uses
local Fisher curvature, not exact nonlinear state-posterior curvature.

---

## Adding a check

Anything that introduces a new numerical approximation needs both layers:

1. a readable reference script in `gbmtoolbox/dev/`, stating the question, the
   reference calculation and the failure criterion; and
2. an automated test in `tests/` that keeps it honest.

For changes to likelihoods, Hessians, filtering, evidence, BMS or HBI, supply a
mathematical reference case. Comparing new code against old code is not
validation.

## Contributing

Before opening a pull request: run `python -m pytest -q`, run the relevant
scripts in `gbmtoolbox/dev/`, add tests for numerical or API changes, and keep
the tutorials in `gbmtoolbox/examples/` short and standalone.

Numerical rules the implementation holds itself to:

- MAP search is multi-start L-BFGS-B.
- The quasi-Newton matrix maintained by L-BFGS-B is never used for inference.
- The observed Hessian is recomputed independently at the final MAP.
- Observed-Hessian eigenvalues are never clipped.
- A finite MAP and a valid Laplace approximation are separate outcomes.
- Discrete-state filtering is approximate and must stay labelled as such.
- Modeller programming errors should surface, not be silently absorbed.



