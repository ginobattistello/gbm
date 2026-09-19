# Changelog

## Unreleased

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
  `lambda ~ Gamma(nu0/2, nu0*s0**2/2)` on the observation precision,
  moment-matched onto `log_observation_sd`. The default prior assumes an
  observation SD near 1, which biases the estimate when the outcome is on a
  different scale and trials are few; this version is scale invariant. The
  scale is estimated robustly with `robust_scale` (`1.4826 * MAD`), which
  resists the outliers that would inflate a standard deviation. Also exported:
  `robust_scale`. See `examples/09_observation_only.py`.

### Changed (breaking: alters Gaussian log-evidence)

- Gaussian models no longer require `observation_covariance`. Omitting it now
  estimates a diagonal $R$ instead of raising. This follows VBA in estimating
  measurement noise automatically, while keeping a log-normal parameterisation
  that is compatible with the Laplace approximation.
- **The default observation-noise prior is now VBA's `Ga(1, 1)` prior on
  observation precision, moment-matched onto `log_observation_sd`**
  (`N(0.2886, 0.4112)`, a 95% prior interval of about `[0.38, 4.69]` on the
  observation SD; matching moments does not preserve quantiles, so the exact
  `Ga(1, 1)` interval is `[0.52, 6.29]`). It replaces the earlier `N(0, 2)`,
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

- Every function and class in the library now carries a docstring (161 of 161,
  up from 32 of 123 public items), including the likelihood primitives in
  `families.py` and the MAP/Laplace pipeline in `optimization.py`.
- `docs/MANUAL.md` gains a section on models with no latent dynamics and on
  `observation_noise_correction`, and its tutorial list now covers all ten.
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
