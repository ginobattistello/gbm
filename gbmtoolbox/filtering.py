"""Unified generalized Gaussian filtering for GBM Toolbox."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .families import get_family
from .linalg import solve_spd, symmetrize


def _generalized_update(model, family, m_pred, P_pred, phi, rho_r, u_t, y_t, *, max_iter: int, tol: float, damping: float, jitter: float):
    """EKF for Gaussian outcomes; support-safe Fisher update otherwise.

    Discrete updates use x = m_pred + P_pred @ s, retaining the support of a
    semidefinite covariance without eigenvectors, square roots or inverses of P.
    """
    n = m_pred.shape[0]
    eye = jnp.eye(n, dtype=m_pred.dtype)

    def obs_fn(x):
        """Observation predictor as a function of the latent state alone."""
        return jnp.atleast_1d(model.observation(x, phi, u_t))

    eta_pred = obs_fn(m_pred)
    if model.family == "gaussian":
        # Ordinary EKF: all likelihood/covariance quantities use the same
        # linearization at the predicted state. R>0 makes S positive definite
        # even when P_pred is singular or has repeated eigenvalues.
        J = jax.jacfwd(obs_fn)(m_pred)
        R = model.observation_covariance_jax(phi, rho_r, u_t, eta_pred.shape[0])
        S = symmetrize(J @ P_pred @ J.T + R)
        K = solve_spd(S, J @ P_pred).T
        residual = jnp.atleast_1d(y_t) - eta_pred
        mode = m_pred + damping * (K @ residual)
        # Joseph form preserves PSD and uses the configured damped gain.
        gain = damping * K
        A = eye - gain @ J
        covariance = symmetrize(A @ P_pred @ A.T + gain @ R @ gain.T)
        log_predictive = family.log_prob(y_t, eta_pred, S)
        return mode, covariance, log_predictive, family.mean(eta_pred, R), jnp.array(1), jnp.array(True)

    def terms(s):
        """Score and Fisher information of the current state estimate."""
        x = m_pred + P_pred @ s
        eta = obs_fn(x)
        score_eta = family.score(y_t, eta)
        W = family.fisher(eta)
        J = jax.jacfwd(obs_fn)(x)
        H = symmetrize(J.T @ W @ J)
        score = J.T @ score_eta
        return x, eta, H, score

    def body(i, carry):
        """One Fisher-scoring iteration of the local Laplace update."""
        s, converged, _ = carry

        def update(s):
            """Damped Fisher-scoring step on the support coordinates ``s``."""
            x, _, H, score = terms(s)
            matrix = (1.0 + jitter) * eye + H @ P_pred
            candidate = s + jnp.linalg.solve(matrix, score - s)
            s_new = s + damping * (candidate - s)
            step = P_pred @ (s_new - s)
            criterion = jnp.linalg.norm(step) <= tol * (1.0 + jnp.linalg.norm(x + step))
            return s_new, criterion, jnp.asarray(i + 1, dtype=jnp.int32)

        return jax.lax.cond(converged, lambda _: carry, update, s)

    s0 = jnp.zeros((n,), dtype=m_pred.dtype)
    s, converged, n_iter = jax.lax.fori_loop(0, max_iter, body, (s0, jnp.array(False), jnp.array(0, dtype=jnp.int32)))
    x_mode, eta_mode, H, _ = terms(s)
    matrix = (1.0 + jitter) * eye + P_pred @ H
    P_filt = symmetrize(jnp.linalg.solve(matrix, P_pred))
    sign, logdet = jnp.linalg.slogdet(matrix)
    logdet = jnp.where(sign > 0, logdet, jnp.nan)
    ll_mode = family.log_prob(y_t, eta_mode)
    log_predictive = ll_mode - 0.5 * (s @ P_pred @ s) - 0.5 * logdet

    # One-step-ahead response prediction is evaluated at the predicted state.
    prediction = family.mean(eta_pred)
    return x_mode, P_filt, log_predictive, prediction, n_iter, converged


def nonlinear_state_filter_jax(model, parameters, prepared_data, *, max_update_iter: int, update_tol: float, damping: float, jitter: float):
    """Run the unified Gaussian/Fisher-Laplace state filter.

    Gaussian observations reduce to the standard linear-Gaussian update when
    the evolution/observation functions are linear. Bernoulli and categorical
    observations use the same posterior-mode update with family-specific score
    and Fisher information. ``max_update_iter`` applies to those discrete
    updates only; the Gaussian update is the closed-form EKF step.
    """
    theta, phi, rho_q, rho_r = model.unpack_parameters(parameters)
    family = get_family(model.family)

    y = prepared_data["y"]
    u = prepared_data["u"]

    m0 = jnp.asarray(model.initial_state, dtype=jnp.float64)
    P0 = model.initial_covariance_jax()

    def one_step(carry, y_t, u_t):
        """Predict, then update, for a single trial."""
        m_pred, P_pred = carry
        m_filt, P_filt, ll, prediction, n_iter, converged = _generalized_update(
            model, family, m_pred, P_pred, phi, rho_r, u_t, y_t, max_iter=max_update_iter, tol=update_tol, damping=damping, jitter=jitter
        )

        def transition(xx):
            """Evolution map as a function of the latent state alone."""
            return jnp.asarray(model.evolution(xx, theta, u_t, y_t), dtype=jnp.float64).reshape(m_filt.shape)

        F = jax.jacfwd(transition)(m_filt)
        next_mean = transition(m_filt)
        Q = model.process_covariance_jax(theta, rho_q, u_t)
        next_cov = symmetrize(F @ P_filt @ F.T + Q)
        outputs = (m_filt, P_filt, m_pred, P_pred, prediction, ll, n_iter, converged)
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

    filt_mean, filt_cov, pred_mean, pred_cov, predictions, loglik, n_iter, converged = outputs
    return {
        "loglik": loglik,
        "states": filt_mean,
        "state_covariance": filt_cov,
        "predicted_state": pred_mean,
        "predicted_covariance": pred_cov,
        "prediction": predictions,
        "update_iterations": n_iter,
        "update_converged": converged,
    }
