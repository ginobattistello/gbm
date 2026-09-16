import jax.numpy as jnp
import numpy as np
from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def test_filtered_bernoulli_supports_ad_hessian():
    def evolution(x, theta, u_t, y_t):
        return x
    def observation(x, phi, u_t):
        return x[0] + phi[0]
    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(GaussianPrior([0.0], [0.0], names=["dummy"]), GaussianPrior([0.0], [1.0], names=["bias"])),
        initial_state=[0.0], initial_state_covariance=[1.0], process_covariance=[0.1],
    )
    y = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1] * 2)
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=1, maxiter=100, verbose=False, filter_max_iter=5))
    assert fit.math.diagnostics[0].laplace_valid
    assert np.isfinite(fit.output.log_evidence[0])


def test_filtered_categorical_supports_ad_hessian():
    def evolution(x, theta, u_t, y_t):
        return x
    def observation(x, phi, u_t):
        return jnp.array([x[0] + phi[0], 0.0, -x[0]])
    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="categorical",
        priors=Priors(GaussianPrior([0.0], [0.0], names=["dummy"]), GaussianPrior([0.0], [1.0], names=["bias"])),
        initial_state=[0.0], initial_state_covariance=[1.0], process_covariance=[0.1],
    )
    y = np.array([0, 1, 2, 0, 0, 1, 2, 0, 1, 0] * 2)
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=1, maxiter=100, verbose=False, filter_max_iter=5))
    assert fit.math.diagnostics[0].laplace_valid
    assert np.isfinite(fit.output.log_evidence[0])
