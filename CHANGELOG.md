# Changelog


## Unreleased (0.2.0.dev0)

### Single-subject vector input (2026-10-13)

- `individual_fit(y, model, u=u)` fits one subject from its outcome vector,
  with trial inputs as a keyword: an array of any dimension whose first axis is
  trials, a dict of such arrays, or `None`. It is wrapped into the usual
  subject dict before validation, so the fit is identical to
  `individual_fit([{"y": y, "u": u}], model)` and no slower. Stacked
  subjects-by-trials matrices are rejected as ambiguous.

### Consistency review (2026-10-13)

Removed (all unused, or kept only for an earlier interface):

- `GroupBMS` and its dictionary output. `bms`, `bms_conditions` and
  `bms_groups` cover the same analyses; pass `alpha0` explicitly for VBA's
  one-total-count prior.
- `rts_smoother_jax` and the filter's `transition_jacobian` output, which only
  fed it; `psd_factor`; `StateModel.split_parameters`;
  `Priors.theta_slice`/`phi_slice`; `FamilyOps.sample`; and the
  `DEFAULT_OBSERVATION_NOISE_PRIOR_MEAN`/`_VARIANCE` constants (use
  `default_observation_noise_prior(1).mean`/`.covariance`).
- Short `hard_bounds` lists covering only `[theta, phi]`. One entry per
  resolved parameter, including estimated noise log-SDs, is now required.

Changed:

- `observation_noise_prior_from_scale` returns the exact conjugate
  `GammaPrecisionPrior(nu0/2, nu0*s0**2/2)` instead of its Gaussian moment
  match, so `prior_scale=1, prior_sample_size=2` is the default prior. An
  outcome with no scale information raises instead of centring at zero.
- Standalone evaluation (`model.evaluate`, `model(...)`, `evaluate_jax`) uses
  the `Config` filter defaults (4 update iterations) instead of 8.
- `noise_parameter_correlations` returns noise rows against free-parameter
  columns (`noise_names`, `parameter_names`); it previously returned the whole
  correlation matrix.
- `observation_noise_correction` returns `valid=False` for models with a
  stochastic latent state, where residuals also contain state variance.
- `convergence_diagnostics` reports zero successful starts when every
  parameter is fixed.
- `search_log_joint` is always recorded, from evaluations already made, so
  `fit.plot()` shows the objective path without `Config(display=True)`.
- The subject field `u` is optional in `individual_fit`, as elsewhere.
- Importing the toolbox sets the JAX compilation cache directory only when the
  user has not configured one; `x64` is enabled once, in `__init__`.
- `prior_predictive` draws every prior through one path, so seeded draws for
  models without a Gamma noise prior differ from earlier versions.
- `Config(parameter_scaling=...)` with a non-boolean raises `TypeError`;
  `HBIConfig.maxiter` must be an integer; `prior_sensitivity` takes an optional
  `config` like the other entry points.
- `dev/13_fit_performance.py` times the default unscaled search; `--scaled`
  replaces `--unscaled`.

Housekeeping: every library function and class has a docstring again; `ruff
check` and `ruff format` are clean (the generated VBA notebook is excluded);
`docs/VALIDATION.md` has one numbered section per reference script (1-15) and
no longer cites results files that are not in the repository; the tests from
`test_priority_fixes.py` moved into topic files, with new `test_hbi.py` and
`test_reproducibility.py`.

### Reliability and reproducibility fixes (2026-10-13)

- Consistent Gaussian EKF and support-preserving discrete filter solves without
  state covariance eigenvector derivatives.
- Require optimizer convergence and positive curvature for Laplace evidence;
  prefer converged starts and reject invalid subject fits in HBI.
- Include covariance in HBI stopping, support empty scientific parameter blocks,
  and keep returned priors aligned with returned fits.
- Split subject optimization/reporting RNG streams; activate HBI seeds and inherit
  the fit seed when no HBI config is supplied.
- Marginalize exactly uninformative BMS rows. Default to marginal family evidence;
  integrate two-family posteriors and two-category group evidences by quadrature.
  Retain explicit aggregate mode and document remaining mean-field limitations.
- Return complete right-singular-vector bases for local identifiability; fix the
  Bernoulli family sampler's probability argument; correct repository metadata.
- Parameter scaling remains disabled by default.


### Extended BMS (2026-10-13)

- Add family inference with balanced default prior mass, exact Dirichlet count
  aggregation, analytic family mean frequencies, and conditional within-family
  analyses.
- Add explicit fixed-profile family null evidence and corresponding family
  BOR/PXP; do not apply an ad hoc model-to-family BOR correction.
- Add paired-condition tuple inference, optional factorial condition axes,
  per-condition reports, and tuple-space allocation guards.
- Add independent-group comparisons using pooled versus separate RFX bounds,
  supporting unequal sample sizes, multiple groups, and configurable hypothesis
  prior odds. Family comparisons normalize within-family evidence mixtures;
  `group_level="models"` compares the full model-frequency profiles instead.
- Dispatch these inputs from `bms`; export `bms_conditions`, `bms_groups`,
  and result dataclasses. (A dictionary-style `GroupBMS` adapter added here was
  removed in the consistency review.)
- Keep the original 2D model-only prior and result fields. Two-candidate EP is
  now analytic; higher-dimensional EP uses seeded, memory-bounded sampling.
- Add analytic Dirichlet/Beta/mixture references, extended regression tests,
  a scientific reference script, and an end-to-end example.
- Document corrected group semantics, family-null assumptions, and differences
  from the supplied script and VBA source. Preserve disabled parameter scaling.

### Fitting performance (2026-10-13)

- Vectorize deterministic observation-only trials with `jax.vmap`; dynamic or
  uncertain states retain sequential evaluation/filtering.
- Use normalized elementwise Gaussian likelihoods for scalar/diagonal noise.
- Share compiled objectives across compatible models, passing Gaussian and
  exact Gamma prior values dynamically. Reuse the optimizer executable for
  preflight validation and cache full prediction/propagation evaluators.
- Prepare subject data once per fit and reuse warmed objective results.
- Add `Config(parameter_scaling=True)`: Cholesky scaling without bounds,
  diagonal scaling with bounds. Returned Hessians, covariance, gradients,
  search paths, and evidence use original parameter coordinates.
- Record stage timings and per-start SciPy evaluation counts and runtimes.
- Add analytic/equivalence tests and `dev/13_fit_performance.py` for cold/warm
  timing measurements.
- Fix three existing Gamma-noise inconsistencies: missing fit-input prior
  metadata, Gaussian-proxy prior-predictive sampling, and proxy rather than
  exact local prior curvature in the information diagnostic. These do not
  change the fitting prior or evidence formula.
- Document execution paths, cache assumptions, scaling effects on convergence,
  numerical invariance checks, and impact on BMS/HBI in the manual.

### Added

- Observation-only models. `StateModel.evolution` and `Priors.evolution` are
  now optional: omit both and `theta` becomes an empty vector while the latent
  state stays fixed. Previously such a model had to declare a
  dummy evolution parameter with zero prior variance, which left an unused
  entry in the parameter layout and the reported summary. Declaring an
  evolution prior without an evolution function is now an error.
- `GaussianPrior.empty()` for a prior over zero parameters.
- `StateModel.initial_state` is now optional, defaulting to `[0.0]`. An
  observation-only model whose observation function ignores `x` gives the same
  fit whatever the initial state holds, so it no longer has to be stated. It is
  still carried internally because the trial recursion needs a state to thread
  through; set it when the observation function actually reads `x`.
- `observation_noise_correction(fit, subject)`, a diagnostic reporting the
  observation SD corrected for posterior parameter uncertainty. The MAP noise
  estimate uses the residuals at the fitted parameters and so reproduces the
  biased maximum-likelihood variance `SSE/T`; the diagnostic adds the expected
  prediction uncertainty `sum_t J_t Sigma J_t^T` under the Laplace posterior.
  With 5 free parameters at 25 trials the uncorrected SD is biased low by
  about 10%. The Jacobian is differentiated through the model's own
  trajectory, so this covers dynamical models as well as static ones. It
  changes neither the fit nor its log-evidence. Validated against the analytic
  result for a linear model in `dev/11_observation_noise_correction.py`.
- `observation_noise_prior_from_scale(y_or_prior_scale, prior_sample_size=2.0)`,
  an opt-in observation-noise prior stated as an expected noise scale `s0` and
  a prior sample size `nu0`. It is the conjugate prior
  `lambda ~ Gamma(nu0/2, nu0*s0**2/2)` on the observation precision (exact
  since the consistency review; first released as a moment match). The default
  prior assumes an observation SD near 1, which biases the estimate when the
  outcome is on a different scale and trials are few; deriving `s0` from the
  data makes the prior follow the outcome's units. The
  scale is estimated robustly with `robust_scale` (`1.4826 * MAD`), which
  resists the outliers that would inflate a standard deviation. Also exported:
  `robust_scale`. See `examples/09_observation_only.py`.

### Changed (breaking: alters Gaussian log-evidence)

- Gaussian models no longer require `observation_covariance`. Omitting it now
  estimates a diagonal $R$ instead of raising. This follows VBA in estimating
  measurement noise automatically, while keeping a log-normal parameterisation
  that is compatible with the Laplace approximation.
- **The default observation-noise prior is now VBA's `Ga(1, 1)` prior on
  observation precision.** It was first introduced as a Gaussian moment match
  on `log_observation_sd` (`N(0.2886, 0.4112)`) and is now evaluated exactly in
  that coordinate, including the change-of-variables Jacobian; its central 95%
  interval on the observation SD is `[0.52, 6.29]`. It replaces the earlier `N(0, 2)`,
  whose implied interval `[0.06, 16]` was roughly five times wider in
  variance. Gaussian log-evidence values therefore shift again; model
  comparisons must be recomputed within one version.
  The prior is Gamma because that is the conjugate form VBA uses, but it is
  still *fitted* on the log scale: GBM Toolbox estimates noise inside one joint
  MAP/Laplace vector, and the exact noise posterior is far less skewed in
  `log(sigma)` than in `sigma^2` (about `-0.28` against `+1.29` at 25 trials),
  so the Gaussian Laplace approximation is much better there. `sigma^2` would
  only be preferable under a variational scheme that keeps precision as a
  separate conjugate factor, as VBA does.
- **Gaussian log-evidence values shift** relative to 0.1.0 for models that
  previously passed a fixed `R`, because the noise scale is now integrated over
  rather than asserted. Model comparisons must be recomputed within one
  version; passing `observation_covariance` explicitly restores the old
  behaviour exactly.
- `observation_noise_prior` is now optional whenever noise is estimated.
- Preflight error for a mismatched estimated-$R$ dimension now names the
  subject and the required `observation_dim`.

### Documentation and housekeeping

- Every function and class in the library now carries a docstring (up from
  32 of 123 public items), including the likelihood primitives in
  `families.py` and the MAP/Laplace pipeline in `optimization.py`.
- `docs/MANUAL.md` gains a section on models with no latent dynamics and on
  `observation_noise_correction`, and its tutorial list covers every tutorial.
- `docs/VALIDATION.md` gains section 11 for the observation-noise correction,
  so every reference script has a documented check.
- Reference scripts in `dev/` are numbered consistently with the sections of
  `docs/VALIDATION.md`; `validate_ad_derivatives.py` and
  `validate_noise_recovery.py` became `08_ad_derivatives.py` and
  `09_noise_recovery.py`, and the two newer checks moved to 10 and 11.
- `gbmtoolbox/examples/` is a real package, so the tutorials keep working from
  an installed wheel; `gbmtoolbox/dev/` is excluded from the wheel instead.
- Removed a duplicated Jacobian helper in `diagnostics.py`: both
  `numerical_local_identifiability` and `observation_noise_correction` now
  share one documented `_free_prediction_function`.
- Noise-prior tests moved out of `test_observation_only.py` into
  `test_noise_priors.py`, so each file covers one topic.
- Formatting and import order are enforced by `ruff`; the few deliberate lint
  exceptions are listed with their reasons in `pyproject.toml`.

### Validation

- `gbmtoolbox/dev/10_default_observation_noise.py`: validates the default
  against the closed-form posterior mode (worst relative error `7.1e-09`) and
  quantifies prior shrinkage against the unpenalised MLE.
- `gbmtoolbox/dev/11_observation_noise_correction.py`: checks the
  parameter-uncertainty correction against the analytic maximum-likelihood
  bias for a linear model, with a tolerance in Monte-Carlo standard errors.

## 0.1.0 — initial independent implementation

- New `StateModel` generative API based on evolution and observation mappings.
- Separate Gaussian priors for evolution (`theta`) and observation (`phi`) parameters.
- Generic `y` / `u` subject data convention.
- Multi-start L-BFGS-B MAP search without an outer Gauss-Newton polish.
- Independently recomputed central finite-difference observed Hessian for Laplace inference.
- Centralized preflight validation and categorized runtime sanity tracking.
- Gaussian EKF and Bernoulli/categorical Laplace-Gaussian/Fisher filtering.
- Propagated and filtered latent-state uncertainty.
- Native latent uncertainty shadows in individual-fit plots.
- Convergence, Hessian, prior-information and numerical local-identifiability diagnostics.
- Prior/posterior predictive checks and prior sensitivity.
- Random-effects BMS plus a compact HBI-inspired hierarchical empirical-Bayes/Laplace refitting implementation.



