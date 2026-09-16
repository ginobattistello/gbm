"""Random-effects Bayesian model selection for group studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, gammaln, psi


@dataclass
class BMSResult:
    posterior_parameters: np.ndarray
    model_frequency: np.ndarray
    exceedance_prob: np.ndarray
    bor: float
    protected_exceedance_prob: np.ndarray
    responsibilities: np.ndarray


def dirichlet_exceedance(alpha, n_samples=100000, random_state=42):
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if np.any(alpha <= 0):
        raise ValueError("Dirichlet parameters must be positive")
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(random_state)
    wins = np.zeros(len(alpha), dtype=np.int64)
    remaining = int(n_samples)
    block = 100000
    while remaining:
        n = min(block, remaining)
        draws = rng.dirichlet(alpha, size=n)
        wins += np.bincount(np.argmax(draws, axis=1), minlength=len(alpha))
        remaining -= n
    return wins / int(n_samples)


def _compute_fe(L, alpha, r, alpha0):
    """Variational free energy; L is models x subjects."""
    Elogr = psi(alpha) - psi(np.sum(alpha))
    Sqf = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)) - np.sum((alpha - 1) * Elogr)
    Sqm = -np.sum(r * np.log(np.clip(r, np.finfo(float).tiny, 1.0)))
    ELJ = gammaln(np.sum(alpha0)) - np.sum(gammaln(alpha0)) + np.sum((alpha0 - 1) * Elogr)
    ELJ += np.sum(r * (Elogr[:, None] + L))
    return float(ELJ + Sqf + Sqm)


def _null_fe(L):
    K, N = L.shape
    F0 = 0.0
    for i in range(N):
        z = L[:, i] - np.max(L[:, i])
        g = np.exp(z)
        g /= np.sum(g)
        F0 += np.sum(g * (L[:, i] - np.log(K) - np.log(np.clip(g, np.finfo(float).tiny, 1.0))))
    return float(F0)


def bms(lme, *, alpha0: np.ndarray | None = None, n_samples=100000, random_state=42, tol=1e-6, maxiter=10000):
    """Random-effects Bayesian model selection.

    Parameters
    ----------
    lme : array, shape (subjects, models)
        Subject log model evidences.
    """
    lme = np.asarray(lme, dtype=float)
    if lme.ndim != 2 or not np.all(np.isfinite(lme)):
        raise ValueError("lme must be a finite subjects x models matrix")
    N, K = lme.shape
    if K < 2:
        raise ValueError("BMS requires at least two models")
    alpha0 = np.ones(K) if alpha0 is None else np.asarray(alpha0, dtype=float).reshape(-1)
    if alpha0.shape != (K,) or np.any(alpha0 <= 0):
        raise ValueError("alpha0 must contain one positive value per model")
    alpha = alpha0.copy()
    for _ in range(maxiter):
        log_r = lme + (psi(alpha) - psi(np.sum(alpha)))[None, :]
        log_r -= np.max(log_r, axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= np.sum(r, axis=1, keepdims=True)
        new_alpha = alpha0 + np.sum(r, axis=0)
        if np.linalg.norm(new_alpha - alpha) <= tol:
            alpha = new_alpha
            break
        alpha = new_alpha
    else:
        raise RuntimeError("BMS variational update did not converge")
    freq = alpha / np.sum(alpha)
    xp = dirichlet_exceedance(alpha, n_samples=n_samples, random_state=random_state)
    F1 = _compute_fe(lme.T, alpha, r.T, alpha0)
    F0 = _null_fe(lme.T)
    bor = float(expit(F0 - F1))
    pxp = (1.0 - bor) * xp + bor / K
    return BMSResult(alpha, freq, xp, bor, pxp, r)
