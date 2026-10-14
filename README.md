# GBM Toolbox

**Bayesian generative modelling of brain and behaviour in Python.**

Write a model of how observations arise; GBM estimates its parameters,
quantifies uncertainty, and compares candidate models across participants.
It supports Gaussian, binary and categorical outcomes, with optional latent
state dynamics and state noise. The numerical core uses JAX automatic differentiation.

## Install

Python 3.10 or newer:

```bash
git clone https://github.com/ginobattistello/gbm.git
cd gbm
python -m pip install -e ".[dev]"
```

## Fit a first model

This runnable example fits a straight line and estimates its observation noise:

```python
import numpy as np
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit

rng = np.random.default_rng(42)
u = np.linspace(-1, 1, 100)
y = 0.5 + 1.2 * u + rng.normal(0, 0.3, len(u))

model = StateModel(
    observation=lambda x, phi, u_t: phi[0] + phi[1] * u_t,
    family="gaussian",
    priors=Priors(observation=GaussianPrior([0, 0], [1, 1])),
)
fit = individual_fit(y, model, u=u, config=Config(random_state=42, verbose=False))
print(fit.output.parameters)       # intercept, slope, log observation SD
print(fit.output.log_evidence)     # approximate log model evidence
print(fit.math.diagnostics[0].laplace_valid)
```

Prior covariance values are **variances**, not standard deviations. Gaussian
observation noise is estimated by default under a proper Gamma(1, 1)
shape–rate prior on precision. This prior assumes a scale of order one in the
outcome's units; set a different prior or known covariance when appropriate.

For several participants, pass a list of `{"y": y, "u": u}` dictionaries.
For learning or state-space models, also define `evolution(x, theta, u_t, y_t)`.
The observation function returns a Gaussian **mean**, a Bernoulli **logit**, or
categorical **logits**. GBM applies sigmoid/softmax for discrete outcomes.

## What GBM provides

- **Individual fitting:** multi-start MAP estimation, an independently computed
  observed Hessian, and Gaussian Laplace posterior/evidence approximations.
- **State inference:** deterministic dynamics or approximate filtering; optional
  reports of parameter-induced or conditional state uncertainty.
- **Model comparison:** random-effects BMS for models, families, paired
  conditions and independent groups.
- **Hierarchical fitting:** `hierarchical_fit` learns group priors by empirical-Bayes
  refitting of subjects. It is inspired by CBM/HBI, not the original HBI algorithm.
- **Model checking:** convergence, posterior curvature, local prediction
  sensitivity, residual-energy diagnostics, and predictive simulations.

Current BMS defaults are one total Dirichlet prior count, aggregate family
inference, and model-level group comparisons. To integrate models uniformly
within families, select `family_method="marginal"`; to compare group family
frequencies, select `group_level="families"`. These are different statistical
analyses, not interchangeable display options.

Check diagnostics before interpreting uncertainty or evidence. A positive
Hessian and optimizer convergence do not guarantee a global optimum, accurate
filtering, or an adequate model.

## Learn and validate

- [Manual](docs/MANUAL.md): a reading guide, examples, assumptions and equations.
- [Validation](docs/VALIDATION.md): reference checks, measured results and limits.
- [Tutorials](gbmtoolbox/examples): learning models, state filtering, diagnostics,
  predictive checks, hierarchical fitting and group comparisons.

```bash
python -m pytest -q
```

## Scientific provenance

GBM's methodological lineage includes the CBM/HBI framework of
[Piray et al. (2019)](https://doi.org/10.1371/journal.pcbi.1007043).
See [NOTICE.md](NOTICE.md), [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE).
The toolbox was developed with AI assistance; independent mathematical review,
replication and bug reports are welcome through issues and pull requests.
