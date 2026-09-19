"""Reference simulation for internally estimated diagonal Q and R."""
import jax.numpy as jnp
import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit

TRUE_A = 0.75
TRUE_Q = 0.08
TRUE_R = 0.30


def evolution(x, theta, u_t, y_t):
    return jnp.asarray([theta[0] * x[0]])


def observation(x, phi, u_t):
    return x[0] + phi[0]


model = StateModel(
    evolution=evolution,
    observation=observation,
    family="gaussian",
    priors=Priors(GaussianPrior([TRUE_A], [0.0], names=["a"]), GaussianPrior([0.0], [0.0], names=["offset"])),
    initial_state=[0.0],
    initial_state_covariance=[1.0],
    process_covariance="diagonal",
    process_noise_prior=GaussianPrior([np.log(np.sqrt(TRUE_Q))], [1.0]),
    observation_covariance="diagonal",
    observation_noise_prior=GaussianPrior([np.log(np.sqrt(TRUE_R))], [1.0]),
    observation_dim=1,
)

rng = np.random.default_rng(8)
T = 300
x = 0.0
y = np.zeros(T)
for t in range(T):
    y[t] = x + rng.normal(scale=np.sqrt(TRUE_R))
    x = TRUE_A * x + rng.normal(scale=np.sqrt(TRUE_Q))

fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=3, verbose=False, latent_uncertainty="filtered"))
q_hat = fit.output.process_noise_sd[0, 0] ** 2
r_hat = fit.output.observation_noise_sd[0, 0] ** 2
print(f"true Q={TRUE_Q:.4f}, estimated Q={q_hat:.4f}")
print(f"true R={TRUE_R:.4f}, estimated R={r_hat:.4f}")
print("posterior correlation matrix includes joint Q/R curvature:")
print(fit.math.covariance[0])
