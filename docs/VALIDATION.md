# Validation

This guide states **what was checked, against which reference, and what the
result does not establish**. Equations and user guidance are in the
[MANUAL](MANUAL.md).

## Reading guide

- [Reproduce the checks](#reproduce-the-checks)
- [Recorded run](#recorded-run)
- [Scientific reference cases](#scientific-reference-cases)
- [Interpretation of the main checks](#interpretation-of-the-main-checks)
- [Coverage limits and release checks](#coverage-limits-and-release-checks)

## Reproduce the checks

From the repository root after `python -m pip install -e ".[dev]"`:

```bash
# Separate processes bound memory from accumulated JAX compilations.
python -m pytest -q --ignore=tests/test_predictive_bms_hbi.py
python -m pytest -q tests/test_predictive_bms_hbi.py
# Independent processes for the scientific reference scripts.
for f in gbmtoolbox/dev/*.py; do python "$f" || exit 1; done
```

The automated tests check numerical references and API contracts. Scripts in
`gbmtoolbox/dev/` expose mathematical reference cases, approximation errors,
or descriptive results. Script completion alone is not scientific validation:
09 reports one recovery realization without an acceptance assertion, and 13
reports timings. Script 06 checks a sampling identity without calling GBM's
propagation API. There is no CI configuration in this reviewed repository;
these are local commands, not evidence of a hosted CI run.

## Recorded run

The replacement files target base commit
`007c890bf3b32c744f9d15304fa592394a43b114` plus the accompanying documentation,
validation-dimension fix, renamed-API callers and explicit-method test updates.
The base commit alone is **not** the tested replacement state.

| Item | Value |
|---|---|
| Date | 2026-10-14 |
| Platform | Linux x86_64, CPU |
| Python | 3.12.14 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| JAX / jaxlib | 0.11.2 / 0.11.2 |
| Automated tests | Broad run: 180 passed, 1 failed; follow-up: 51 passed (overlapping subset, including four added regressions and the three separately run predictive/BMS/hierarchical tests) |
| Reference scripts | 15/15 completed: 13 assertion-bearing scripts passed; 09 and 13 are descriptive |

The follow-up command was
`python -m pytest -q tests/test_predictive_bms_hbi.py tests/test_validation.py tests/test_bms_extended.py`.
A further exact residual-energy unit test is recorded with the supplied logs.
Do not sum overlapping run counts.

**Remaining failure:** `test_pickled_model_recomputes_its_execution_key` in
`tests/test_reproducibility.py`. A pickled `StateModel` retains its cached
`execution_key`, which contains callable identities. This separate serialization
issue is not repaired here; do not describe this revision as a fully green
release. `state_model.py` and `model_selection.py` are unchanged.

The previous document's “174 passed” result was dated 2026-10-13 without an
exact tested commit. It is not carried forward as evidence for these files.
In the unmodified base, test collection had five obsolete-API import errors;
a targeted run had seven failures related to obsolete dimension access and
BMS expectations. The replacement updates callers/expectations to the current
public API/defaults; it does not change model-selection algorithms or defaults.

## Scientific reference cases

All script names below are relative to `gbmtoolbox/dev/`. Errors are from the
recorded replacement run, not universal accuracy guarantees. “Tolerance”
refers to the script's actual assertion, including relative tolerance where used.

| No. / script | Reference and acceptance check | Observed result / scope |
|---|---|---|
| 01 `01_likelihood_reference_cases.py` | Elementary normalized Gaussian, Bernoulli and categorical likelihoods; error below `1e-12` | Zero error in these deterministic cases |
| 02 `02_hessian_failure_cases.py` | Hessian `diag(1, epsilon)` at zero; currently asserts positive smallest eigenvalue | Recovered `epsilon=1e-10` through `1e-14`; conditions `1e10` through `1e14`. This script alone does not test optimizer validity flags |
| 03 `03_linear_gaussian_filter.py` | Independent scalar Kalman recursion; `allclose(atol=1e-7)` with NumPy's default relative tolerance | Mean `2.78e-17`, variance `5.55e-17`, log likelihood `1.11e-16` maximum absolute errors |
| 04 `04_bernoulli_filter_quadrature.py` | One-step Gaussian-state probability integral; absolute error below `0.04` | Error `1.231e-3`; one affine-logit case |
| 05 `05_categorical_filter_quadrature.py` | One-step three-category Gaussian-state integral; absolute error below `0.06` | Error `4.136e-3`; one affine-logit case |
| 06 `06_propagated_uncertainty.py` | `Var(a theta)=a² Var(theta)`; sampling within `rtol=0.015` | Relative error `0.005518`; checks the sampling principle, not end-to-end GBM propagation |
| 07 `07_information_identifiability.py` | Binary logit `phi0+phi1`; prediction-Jacobian rank/nullity must be `1/1` | Recovered null direction proportional to `(1,-1)` |
| 08 `08_ad_derivatives.py` | Autodiff versus central differences; MAP `rtol=1e-5, atol=1e-6`, Hessian `rtol=atol=2e-3` | MAP difference zero; maximum Hessian difference `1.542e-4` |
| 09 `09_noise_recovery.py` | Simulated `Q=0.08`, `R=0.30`; descriptive, no recovery assertion | Estimates `Q=0.0926`, `R=0.3123`; not a bias/coverage or identifiability study |
| 10 `10_default_observation_noise.py` | Exact conditional Gamma log-SD mode; relative error below `1e-3` | Maximum relative mode error `3.342e-8` |
| 11 `11_observation_noise_correction.py` | Exact linear-Gaussian posterior residual energy; chi-square OLS SD sampling moments | Maximum residual-energy error `1.7e-9`; maximum Monte Carlo discrepancy `0.574` SE |
| 12 `12_gamma_precision_evidence.py` | Analytic Gamma integral and known Laplace error; also a free-intercept quadrature reference | Mode error `7.563e-8`, relative Hessian error `1.513e-7`, predicted-evidence error `7.563e-8`; free-intercept evidence error `-0.02168` |
| 13 `13_fit_performance.py` | Cold/warm runtime, no mathematical acceptance criterion | Timings only; no cross-version speedup claim |
| 14 `14_bms_families_conditions_groups.py` | Dirichlet integral with effectively known labels, Beta EP, fixed family-null mixture and tuple evidence; errors below `1e-10` | Maximum reference error `9.770e-14` |
| 15 `15_reliability_reproducibility.py` | Independent binary frequency integrals and multistate Gaussian filter finite differences | BMS errors at most `4.580e-16`; gradient `2.676e-9`, Hessian `2.077e-8` discrepancies |

## Interpretation of the main checks

### Gaussian filtering and jitter

At unit damping, linear-Gaussian filtering reduces to the Kalman recursion.
The current Gaussian branch does **not** use `Config.filter_jitter`: its gain
and predictive likelihood solve with the innovation covariance directly.
The old jitter-dependent error table described an earlier implementation and
has been removed. Current discrepancies in reference 03 are roundoff-sized.
For nonlinear Gaussian models, both observation and transition linearizations
can introduce approximation error regardless of jitter.

`filter_jitter` regularizes discrete-state-update solves and determinant terms;
it is not process noise. References 04–05 compare `exp(loglik)` with integrated
probabilities, not the plug-in `prediction` field. They test one observed outcome
in each selected state distribution, not normalization over all outcomes,
long-run calibration, or nonlinear/multimodal state inference.

### Local prediction-sensitivity diagnostic

Reference 07 tests `numerical_local_identifiability`, whose public name is
retained. The calculated rank concerns first-order changes in reported
predictions. A parameter that changes only variance can have a zero prediction
Jacobian while remaining identifiable from the likelihood. Neither the script
nor the diagnostic proves global structural identifiability.

### Residual-energy diagnostic, not unbiased noise correction

Reference 11 now separates two questions:

1. With known Gaussian observation variance and a Gaussian coefficient prior,
   does GBM's diagnostic equal the analytic posterior expectation of squared
   residuals? The reference computes the exact coefficient mean/covariance,
   then `SSE + tr(X Sigma X.T)`. The trace tolerance is `1e-10`; corrected
   variance uses `rtol=1e-5, atol=1e-6` to allow MAP optimization error.
2. In the OLS limiting example with full-rank design and true SD one, do sampled
   residual and one-step SDs match their exact chi-square sampling means?
   The check uses 200,000 draws per case and five Monte Carlo standard errors.

For `nu=T-p`, the exact expected residual SD is

$$E[\sqrt{SSE/T}]=\sigma\sqrt{2/T}
\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)}.$$

The one-step multiplier is $\sqrt{1+p/T}$. Its variance estimate has expectation
$\sigma^2[1-(p/T)^2]$, not $\sigma^2$. At `T=25,p=5`, that expected variance is
`0.96 sigma²`. The unbiased variance `SSE/(T-p)` arises from a different
self-consistency calculation; its square root is itself biased.

These checks establish the residual-energy calculation and the stated linear
reference, **not unbiasedness of the default Gamma-prior noise estimate**.
The former script's broad log-SD Gaussian-prior recovery experiment and its
approximate SD bias factor have been replaced, not relabeled as exact evidence.
The diagnostic is read-only; it does not alter the fit, posterior or evidence.

### Gamma prior and evidence

References 10 and 12 use fixed predictions, where precision has exact posterior
`Gamma(A,B)`, `A=a+T/2`, `B=b+SSE/2`. Reference 12 checks the log-SD mode
(error below `2e-6`), Hessian (relative error below `5e-6`), and Laplace evidence
against its analytically predicted approximation error (below `2e-6`).
It checks a log Bayes factor to `2e-6`, BMS common-offset invariance to `1e-12`,
and zero-residual evidence finiteness. In the Bayes-factor case the same `A`
makes the one-dimensional Laplace errors cancel; this is not general BF exactness.

The additional free-intercept example integrates precision analytically and
integrates the intercept numerically over `[-10,10]` with a standard Gaussian
prior. Its joint Laplace log-evidence error must be below `0.1` in that case.
These examples do not certify arbitrary nonlinear or stochastic-state evidence.
`tests/test_gamma_precision.py` also checks density transformation, moments,
prior draws, independent multivariate precisions and unit changes.

### Model selection and current defaults

The default is one total prior count, `family_method="aggregate"`, and
`group_level="models"`. Tests explicitly select `family_method="marginal"`
when checking integrated two-family inference, and `group_level="families"`
when checking shared family-frequency hypotheses. Separate tests lock down
the actual defaults. The numerical implementation of `model_selection.py`
is unchanged by this documentation revision.

Reference 14 checks solvable limits. Reference 15 checks weak evidence with
**explicit marginal family inference**, not the aggregate default: BMS errors
must be below `1e-10`, filter gradient discrepancy below `1e-6`, and Hessian
discrepancy below `2e-5`. The latter is a multistate Gaussian filter check;
it does not independently validate nonlinear discrete-state curvature.

Binary group odds use frequency integration even when model-only summary
objects use variational inference. Two marginal families use integration;
aggregate family reports and larger spaces remain approximate. Differences
of variational bounds can have non-cancelling errors. Strong evidence or
accurate numerical integration of a supplied evidence matrix does not repair
inaccurate subject-level model evidence.

## Coverage limits and release checks

Existing tests cover fixed parameters, likelihoods, Hessians, covariance paths,
parameter scaling, compilation reuse, seeded fitting, BMS hypotheses and
hierarchical output consistency. The dimension regression additionally checks
scalar and multivariate default noise and an actionable error for a mismatch.
Hierarchical tests use `hierarchical_fit` and `HierarchicalFitConfig`; their
historical filenames retain `hbi`.

Important limits remain:

- A converged outer optimizer and positive Hessian do not imply inner discrete
  filter convergence. Check `update_converged` and stability to iteration settings.
- No general posterior-coverage or model-recovery guarantee follows from the
  selected reference cases or one simulated noise-recovery dataset.
- Full-data state smoothing and total state-plus-parameter uncertainty are not
  provided by the filtered/propagated reporting options.
- Hierarchical tests check empirical-Bayes updates and numerical consistency,
  not a full HBI variational bound or integrated group-hyperparameter posterior.
- Hard-bound evidence is not a normalized truncated-prior calculation.
- Prior-unit information thresholds need qualification for non-Gaussian priors.
- VBA parameter agreement is supplementary replication, not proof of matching
  algorithms or general evidence accuracy. See [comparison notes](../comparison_vba/README.md).

Before release, retain the exact source commit or patch, dependency versions,
commands and outputs; rerun tests and affected reference scripts after code
changes. Update this record rather than copying an earlier pass count. New
likelihood or inference approximations need a mathematical reference and an
automated regression check. Benchmarks should report model/data/settings,
compilation state and hardware separately from scientific accuracy.
