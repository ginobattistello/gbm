"""Reference check 05: categorical Laplace-Gaussian filter versus quadrature."""

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from gbmtoolbox import GaussianPrior, Priors, StateModel


def probs(x):
    z = np.array([x, 0.0, -x])
    e = np.exp(z - z.max())
    return e / e.sum()


model = StateModel(
    lambda x, th, u, y: x,
    lambda x, ph, u: jnp.asarray([x[0], 0.0, -x[0]]),  # logits; families.py applies the softmax
    "categorical",
    Priors(GaussianPrior([0], [0]), GaussianPrior([0], [0])),
    [0.2],
    initial_state_covariance=[0.7],
)
y = 2
out = model.evaluate([0, 0], {"y": np.array([y]), "u": None})
approx = float(np.exp(out["loglik"][0]))
mu = 0.2
var = 0.7
normal = lambda x: np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
exact = quad(lambda x: probs(x)[y] * normal(x), -10, 10, epsabs=1e-12)[0]
print(f"exact predictive probability = {exact:.8f}")
print(f"Laplace-Gaussian approximation = {approx:.8f}")
print(f"absolute error = {abs(approx - exact):.3e}")
assert abs(approx - exact) < 0.06
print("PASS within documented approximation tolerance")
