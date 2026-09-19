import jax.numpy as jnp
import numpy as np

from gbmtoolbox import GaussianPrior, Priors, StateModel


def test_unified_filter_matches_scalar_kalman():
    A, Q, R, P0 = 0.8, 0.1, 0.25, 1.0

    def evolution(x, theta, u_t, y_t):
        return jnp.asarray([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return x[0]

    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(GaussianPrior([A], [0.0]), GaussianPrior([0.0], [0.0])),
        initial_state=[0.0],
        initial_state_covariance=[P0],
        process_covariance=[Q],
        observation_covariance=[R],
    )
    y = np.array([0.2, -0.1, 0.4, 0.0])
    run = model.evaluate(model.parameter_layout.mean, {"y": y, "u": None})
    m, P = 0.0, P0
    ms, Ps, lls = [], [], []
    for yt in y:
        S = P + R
        lls.append(-0.5 * (np.log(2 * np.pi * S) + (yt - m) ** 2 / S))
        K = P / S
        mf = m + K * (yt - m)
        Pf = (1 - K) * P
        ms.append(mf)
        Ps.append(Pf)
        m, P = A * mf, A * A * Pf + Q
    np.testing.assert_allclose(run["states"][:, 0], ms, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(run["state_covariance"][:, 0, 0], Ps, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(run["loglik"], lls, rtol=1e-7, atol=1e-7)
