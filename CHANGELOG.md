# Changelog

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
