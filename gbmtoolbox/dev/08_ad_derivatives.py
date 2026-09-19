"""Reference check: JAX observed Hessian agrees with central finite differences."""
import jax.numpy as jnp
import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def evolution(x, theta, u_t, y_t):
    return jnp.asarray([theta[0] * x[0] + u_t])


def observation(x, phi, u_t):
    return x[0] + phi[0]


rng = np.random.default_rng(3)
T = 40
u = rng.normal(size=T)
x = 0.0
y = np.zeros(T)
for t in range(T):
    y[t] = x + 0.2 + rng.normal(scale=0.4)
    x = 0.7 * x + u[t]

data = [{"y": y, "u": u}]
model = StateModel(
    evolution=evolution,
    observation=observation,
    family="gaussian",
    priors=Priors(GaussianPrior([0.6], [0.5], names=["gain"]), GaussianPrior([0.0], [1.0], names=["offset"])),
    initial_state=[0.0],
    observation_covariance=[0.16],
)
fit_ad = individual_fit(data, model, config=Config(num_init=1, verbose=False, hessian_method="autodiff"))
fit_fd = individual_fit(data, model, config=Config(num_init=1, verbose=False, hessian_method="central_fd"))
print("MAP max abs difference:", np.max(np.abs(fit_ad.output.parameters - fit_fd.output.parameters)))
print("Hessian max abs difference:", np.max(np.abs(fit_ad.math.hessian[0] - fit_fd.math.hessian[0])))
np.testing.assert_allclose(fit_ad.output.parameters, fit_fd.output.parameters, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(fit_ad.math.hessian[0], fit_fd.math.hessian[0], rtol=2e-3, atol=2e-3)
print("PASS")
