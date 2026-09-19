"""Unified generalized Gaussian filtering and smoothing for GBM Toolbox."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .families import get_family
from .linalg import logdet_spd, psd_factor, solve_spd, symmetrize


def _generalized_update(model, family, m_pred, P_pred, phi, rho_r, u_t, y_t, *, max_iter: int, tol: float, damping: float, jitter: float):
    """Posterior-mode/Fisher update in fixed-shape standardized support coordinates."""
    L = psd_factor(P_pred)
    n = m_pred.shape[0]
    eye = jnp.eye(n, dtype=m_pred.dtype)

    def terms(z):
        """Score and Fisher information of the current state estimate."""
        x = m_pred + L @ z

        def obs_fn(xx):
            """Observation predictor as a function of the latent state alone."""
            return jnp.atleast_1d(model.observation(xx, phi, u_t))

        eta = obs_fn(x)
        R = None
        if model.family == "gaussian":
            R = model.observation_covariance_jax(phi, rho_r, u_t, eta.shape[0])
        score_eta = family.score(y_t, eta, R)
        W = family.fisher(eta, R)
        J = jax.jacfwd(obs_fn)(x)
        Jz = J @ L
        A = symmetrize(eye + Jz.T @ W @ Jz)
        g = -z + Jz.T @ score_eta
        return x, eta, R, A, g

    def body(i, carry):
        """One Fisher-scoring iteration of the local Laplace update."""
        z, converged, n_iter = carry
        _, _, _, A, g = terms(z)
        raw_step = solve_spd(A, g, jitter=jitter)
        step = damping * raw_step
        step = jnp.where(converged, jnp.zeros_like(step), step)
        z_new = z + step
        criterion = jnp.linalg.norm(step) <= tol * (1.0 + jnp.linalg.norm(z_new))
        n_iter = jnp.where(converged, n_iter, i + 1)
        converged = jnp.logical_or(converged, criterion)
        return z_new, converged, n_iter

    z0 = jnp.zeros((n,), dtype=m_pred.dtype)
    z, converged, n_iter = jax.lax.fori_loop(0, max_iter, body, (z0, jnp.array(False), jnp.array(0, dtype=jnp.int32)))
    x_mode, eta_mode, R_mode, A, _ = terms(z)
    cov_z = solve_spd(A, eye, jitter=jitter)
    P_filt = symmetrize(L @ cov_z @ L.T)
    ll_mode = family.log_prob(y_t, eta_mode, R_mode)
    log_predictive = ll_mode - 0.5 * (z @ z) - 0.5 * logdet_spd(A, jitter=jitter)

    # One-step-ahead response prediction is evaluated at the predicted state.
    eta_pred = jnp.atleast_1d(model.observation(m_pred, phi, u_t))
    R_pred = None
    if model.family == "gaussian":
        R_pred = model.observation_covariance_jax(phi, rho_r, u_t, eta_pred.shape[0])
    prediction = family.mean(eta_pred, R_pred)
    return x_mode, P_filt, log_predictive, prediction, n_iter, converged


def nonlinear_state_filter_jax(
    model, parameters, prepared_data, *, max_update_iter: int = 4, update_tol: float = 1e-8, damping: float = 1.0, jitter: float = 1e-9
):
    """Run the unified Gaussian/Fisher-Laplace state filter.

    Gaussian observations reduce to the standard linear-Gaussian update when
    the evolution/observation functions are linear. Bernoulli and categorical
    observations use the same posterior-mode update with family-specific score
    and Fisher information.
    """
    theta, phi, rho_q, rho_r = model.unpack_parameters(parameters)
    family = get_family(model.family)
    # Gaussian observations recover the ordinary EKF update with one
    # Fisher/Gauss-Newton step. Iterating is unnecessary unless we later
    # explicitly introduce an iterated-EKF mode.
    if model.family == "gaussian":
        effective_max_iter = 1
    else:
        effective_max_iter = max_update_iter

    y = prepared_data["y"]
    u = prepared_data["u"]

    m0 = jnp.asarray(model.initial_state, dtype=jnp.float64)
    P0 = model.initial_covariance_jax()

    def one_step(carry, y_t, u_t):
        """Predict, then update, for a single trial."""
        m_pred, P_pred = carry
        m_filt, P_filt, ll, prediction, n_iter, converged = _generalized_update(
            model, family, m_pred, P_pred, phi, rho_r, u_t, y_t, max_iter=effective_max_iter, tol=update_tol, damping=damping, jitter=jitter
        )

        def transition(xx):
            """Evolution map as a function of the latent state alone."""
            return jnp.asarray(model.evolution(xx, theta, u_t, y_t), dtype=jnp.float64).reshape(m_filt.shape)

        F = jax.jacfwd(transition)(m_filt)
        next_mean = transition(m_filt)
        Q = model.process_covariance_jax(theta, rho_q, u_t)
        next_cov = symmetrize(F @ P_filt @ F.T + Q)
        outputs = (m_filt, P_filt, m_pred, P_pred, prediction, ll, n_iter, converged, F)
        return (next_mean, next_cov), outputs

    if u is None:

        def step(carry, y_t):
            """scan body: run one trial of the filter."""
            return one_step(carry, y_t, None)

        _, outputs = jax.lax.scan(step, (m0, P0), y)
    else:

        def step(carry, inp):
            """scan body: run one trial of the filter."""
            y_t, u_t = inp
            return one_step(carry, y_t, u_t)

        _, outputs = jax.lax.scan(step, (m0, P0), (y, u))

    filt_mean, filt_cov, pred_mean, pred_cov, predictions, loglik, n_iter, converged, transition_jacobian = outputs
    return {
        "loglik": loglik,
        "states": filt_mean,
        "state_covariance": filt_cov,
        "predicted_state": pred_mean,
        "predicted_covariance": pred_cov,
        "prediction": predictions,
        "update_iterations": n_iter,
        "update_converged": converged,
        "transition_jacobian": transition_jacobian,
    }


def rts_smoother_jax(filtered_mean, filtered_covariance, predicted_mean, predicted_covariance, transition_jacobian, *, jitter: float = 1e-9):
    """Rauch-Tung-Striebel smoothing of the Gaussian filter approximation."""
    if filtered_mean.shape[0] <= 1:
        return filtered_mean, filtered_covariance

    def step(carry, inp):
        """scan body: run one trial of the filter."""
        m_next_s, P_next_s = carry
        m, P, mp_next, Pp_next, F = inp
        G = solve_spd(Pp_next, F @ P, jitter=jitter).T
        m_s = m + G @ (m_next_s - mp_next)
        P_s = symmetrize(P + G @ (P_next_s - Pp_next) @ G.T)
        return (m_s, P_s), (m_s, P_s)

    init = (filtered_mean[-1], filtered_covariance[-1])
    xs = (filtered_mean[:-1], filtered_covariance[:-1], predicted_mean[1:], predicted_covariance[1:], transition_jacobian[:-1])
    _, (m_rev, P_rev) = jax.lax.scan(step, init, xs, reverse=True)
    return (jnp.concatenate([m_rev, filtered_mean[-1][None, :]], axis=0), jnp.concatenate([P_rev, filtered_covariance[-1][None, :, :]], axis=0))
