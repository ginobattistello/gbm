# Changelog

## Unreleased

### Changed (breaking: alters Gaussian log-evidence)

- Gaussian models no longer require `observation_covariance`. Omitting it now
  estimates a diagonal $R$ instead of raising, under a default
  `N(0, 2)` prior on `log_observation_sd` (weakly informative: 95% prior
  interval of roughly `[0.06, 16]` on the observation SD, wider than the
  `Ga(1,1)` precision prior used by the VBA toolbox). This follows VBA in
  estimating measurement noise automatically, while keeping a log-normal
  parameterisation that is compatible with the Laplace approximation.
- **Gaussian log-evidence values shift** relative to 0.1.0 for models that
  previously passed a fixed `R`, because the noise scale is now integrated over
  rather than asserted. Model comparisons must be recomputed within one
  version; passing `observation_covariance` explicitly restores the old
  behaviour exactly.
- `observation_noise_prior` is now optional whenever noise is estimated.
- Preflight error for a mismatched estimated-$R$ dimension now names the
  subject and the required `observation_dim`.

### Added

- `gbmtoolbox/dev/08_default_observation_noise.py`: validates the default
  against the closed-form posterior mode (worst relative error `7.6e-09`) and
  quantifies prior shrinkage against the unpenalised MLE.

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
