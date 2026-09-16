import numpy as np
from gbmtoolbox import GaussianPrior, Priors, StateModel
import jax.numpy as jnp


def test_diagonal_noise_layout_broadcasts_scalar_priors():
    def evolution(x, theta, u_t, y_t):
        return x
    def observation(x, phi, u_t):
        return jnp.asarray([x[0] + phi[0]])
    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(GaussianPrior([0.0], [1.0], names=["theta"]), GaussianPrior([0.0], [1.0], names=["phi"])),
        initial_state=[0.0, 0.0],
        process_covariance="diagonal",
        process_noise_prior=GaussianPrior([-1.0], [0.5]),
        observation_covariance="diagonal",
        observation_noise_prior=GaussianPrior([-0.7], [0.5]),
        observation_dim=1,
    )
    layout = model.parameter_layout
    assert layout.dim == 5
    assert layout.process_noise_slice == slice(2, 4)
    assert layout.observation_noise_slice == slice(4, 5)
    np.testing.assert_allclose(np.diag(layout.covariance)[2:4], 0.5)
