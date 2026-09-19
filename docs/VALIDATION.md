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

## Running the checks

```bash
python -m pytest -q                          # merge gate
for f in gbmtoolbox/dev/*.py; do python "$f"; done   # scientific layer
```

## Environment for the results below

| | |
|---|---|
| Python | 3.12.12 |
| NumPy | 2.4.2 |
| SciPy | 1.17.1 |
| JAX / jaxlib | 0.11.0 |
| Platform | Darwin arm64 |
| Date | 2026-09-16 |

**Automated suite: 51 passed, 0 failed. Reference scripts: 10 of 10 pass.**

CI additionally runs the suite on Python 3.10, 3.11, 3.12 and 3.13.

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

`validate_ad_derivatives.py`

**Question.** The default Hessian is computed by automatic differentiation. Does
it agree with the central finite-difference Hessian it replaced?

**Reference.** The same fit run with `hessian_method="central_fd"`.

**Result.** MAP difference `0.0`; Hessian maximum absolute difference
`1.53e-04`, consistent with finite-difference truncation error rather than a
disagreement between the two methods.

## 9. Noise-parameter recovery

`validate_noise_recovery.py`

**Question.** When process and observation noise are estimated rather than
fixed, are they recovered, and does the posterior reflect their joint curvature?

**Reference.** Data simulated from known $Q = 0.08$ and $R = 0.30$.

**Result.** Recovered $Q = 0.0926$, $R = 0.3123$. The posterior correlation
matrix carries non-zero joint $Q/R$ curvature, confirming the noise
parameters enter the full posterior rather than being profiled out.

## 10. Default observation-noise prior against the analytic posterior mode

`08_default_observation_noise.py`

**Question.** When `observation_covariance` is omitted, a Gaussian model
estimates $R$ under a default $\rho \sim \mathcal{N}(0, 2)$ prior on
$\rho = \log \sigma$. Does the fitted value match what that model implies
analytically, and how far does the prior bend it away from the data?

**Reference.** For a constant-mean Gaussian with the mean fixed, the log
posterior in $\rho$ is closed-form,

$$L(\rho) = -T\rho - \frac{S}{2e^{2\rho}} - \frac{(\rho - m_0)^2}{2v}, \qquad S = \sum_t (y_t - \mu)^2,$$

so the posterior mode solves $(\rho - m_0) + v\,(T - S e^{-2\rho}) = 0$, where
$\mathcal{N}(m_0, v)$ is the default prior — VBA's $\mathrm{Ga}(1,1)$ prior on
observation precision, moment-matched onto the log scale. That root is found
independently by Brent's method; the unpenalised MLE $\sigma^2 = S/T$ is
reported alongside it so the prior's influence is visible.

**Criterion.** Relative error above `1e-3` against the analytic mode fails.

**Result.** Worst relative error `7.1e-09` across four cases spanning
$\sigma \in [0.3, 2.5]$ and $T \in [200, 800]$ — the estimator reproduces the
analytic posterior mode. Shrinkage against the unpenalised MLE is `+3.95%` at
$T = 50$, `+0.94%` at $T = 200$ and `+0.18%` at $T = 1000$: the default prior
is weakly informative and its influence vanishes with data.

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
