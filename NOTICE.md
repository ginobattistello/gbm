# Provenance and attribution

GBM Toolbox is a redesigned software project developed from the Computational
Behavioral Modeling (CBM) Python toolbox and its methodological framework.

Source lineage:

- original CBM Python repository: https://github.com/payampiray/cbm_python
- CBM/HBI methodological reference: Piray et al. (2019),
  DOI: 10.1371/journal.pcbi.1007043

The inherited CBM code is distributed under the MIT License. The original
copyright notice is retained in `LICENSE`.

GBM Toolbox keeps the methodological lineage of individual MAP estimation with a
local Laplace approximation, random-effects Bayesian model selection, and the
hierarchical Bayesian inference framework of Piray et al. It introduces a new
generative state-model API, a separate numerical MAP/observed-Hessian pipeline,
centralized validation, latent-state uncertainty methods, nonlinear filtering,
diagnostics, predictive checks, and new visualization/documentation.

The Bernoulli/categorical state filter is a local Laplace-Gaussian
approximation using Fisher scoring. It is approximate, not exact Bayesian
filtering. The implementation is validated against numerical reference cases
in `gbmtoolbox/dev/` and `tests/`.
