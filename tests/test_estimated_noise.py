import jax.numpy as jnp
import numpy as np

from gbmtoolbox import Config, GaussianPrior, Priors, StateModel, individual_fit


def test_estimated_q_r_are_positive_and_in_full_posterior():
    def evolution(x, theta, u_t, y_t):
        return jnp.asarray([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return x[0] + phi[0]

    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(GaussianPrior([0.7], [0.0]), GaussianPrior([0.0], [0.0])),
        initial_state=[0.0],
        initial_state_covariance=[1.0],
        process_covariance="diagonal",
        process_noise_prior=GaussianPrior([-1.2], [0.5]),
        observation_covariance="diagonal",
        observation_noise_prior=GaussianPrior([-0.7], [0.5]),
        observation_dim=1,
    )
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=1, maxiter=100, verbose=False))
    assert fit.output.process_noise_sd.shape == (1, 1)
    assert fit.output.observation_noise_sd.shape == (1, 1)
    assert fit.output.process_noise_sd[0, 0] > 0
    assert fit.output.observation_noise_sd[0, 0] > 0
    assert fit.output.parameters.shape[1] == 4
    assert fit.math.hessian[0].shape == (2, 2)  # only Q/R are free


def test_theta_phi_only_hard_bounds_remain_valid_with_internal_noise():
    def evolution(x, theta, u_t, y_t):
        return jnp.asarray([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return x[0] + phi[0]

    model = StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(GaussianPrior([0.7], [0.2]), GaussianPrior([0.0], [0.2])),
        initial_state=[0.0],
        observation_covariance="diagonal",
        observation_noise_prior=GaussianPrior([-0.7], [0.5]),
    )
    data = [{"y": np.zeros(6), "u": None}]
    cfg = Config(num_init=1, maxiter=20, verbose=False, hard_bounds=[(-0.99, 0.99), (-2.0, 2.0)])
    fit = individual_fit(data, model, config=cfg)
    assert fit.output.parameters.shape[1] == 3


def test_default_observation_noise_matches_analytic_posterior_mode():
    """Omitted observation_covariance must reproduce the closed-form posterior mode."""
    from scipy.optimize import brentq

    from gbmtoolbox.parameters import DEFAULT_OBSERVATION_NOISE_PRIOR_MEAN, DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE

    mu = 1.25
    m0, v = DEFAULT_OBSERVATION_NOISE_PRIOR_MEAN, DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE
    rng = np.random.default_rng(11)
    y = mu + rng.normal(scale=0.4, size=300)

    model = StateModel(
        evolution=lambda x, theta, u_t, y_t: x,
        observation=lambda x, phi, u_t: mu,
        family="gaussian",
        priors=Priors(GaussianPrior([0.0], [0.0]), GaussianPrior([0.0], [0.0])),
        initial_state=[0.0],
    )
    assert model.observation_covariance_mode == "diagonal"

    fit = individual_fit([{"y": y, "u": None}], model, config=Config(num_init=3, verbose=False))
    fitted = float(fit.output.observation_noise_sd[0, 0])

    # Stationary point of the log posterior in r = log_observation_sd, with a
    # N(m0, v) prior: (r - m0) + v * (T - S * exp(-2r)) = 0.
    T, S = y.size, float(np.sum((y - mu) ** 2))
    reference = np.exp(brentq(lambda r: (r - m0) + v * (T - S * np.exp(-2.0 * r)), -20.0, 20.0))
    assert abs(fitted - reference) / reference < 1e-3
