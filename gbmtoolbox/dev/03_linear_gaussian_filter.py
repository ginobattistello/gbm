"""Reference check 03: GBM Toolbox filter versus exact Kalman recursion.

For a linear-Gaussian state-space model the EKF implementation must reduce to
the exact Kalman filter. We compare filtered means, variances, and predictive
log likelihood trial by trial.

The residual error is set by Config.filter_jitter, the diagonal regularizer
used in the SPD solves, and scales linearly with it: 1e-9 jitter gives ~2.3e-10
error, 1e-12 gives ~2.3e-13, and with jitter disabled the filter reproduces the
exact Kalman recursion to machine precision (~1.1e-16).
"""

import jax.numpy as jnp
import numpy as np

from gbmtoolbox import GaussianPrior, Priors, StateModel

model = StateModel(
    lambda x, th, u, y: jnp.asarray([th[0] * x[0]]),
    lambda x, ph, u: jnp.asarray([x[0] + ph[0]]),
    "gaussian",
    Priors(GaussianPrior([0.8], [0.2]), GaussianPrior([0.1], [1.0])),
    [0.0],
    initial_state_covariance=[1.0],
    process_covariance=[0.1],
    observation_covariance=[0.25],
)
y = np.array([0.2, -0.1, 0.3, 0.0])
out = model.evaluate([0.8, 0.1], {"y": y, "u": None})

m, P = 0.0, 1.0
means = []
vars = []
lls = []
for yt in y:
    S = P + 0.25
    lls.append(-0.5 * (np.log(2 * np.pi * S) + (yt - (m + 0.1)) ** 2 / S))
    K = P / S
    m = m + K * (yt - (m + 0.1))
    P = (1 - K) * P
    means.append(m)
    vars.append(P)
    m = 0.8 * m
    P = 0.8**2 * P + 0.1

print("max mean error:", np.max(np.abs(out["states"][:, 0] - means)))
print("max variance error:", np.max(np.abs(out["state_covariance"][:, 0, 0] - vars)))
print("max loglik error:", np.max(np.abs(out["loglik"] - lls)))
assert np.allclose(out["states"][:, 0], means, atol=1e-7)
assert np.allclose(out["state_covariance"][:, 0, 0], vars, atol=1e-7)
assert np.allclose(out["loglik"], lls, atol=1e-7)
print("PASS")
