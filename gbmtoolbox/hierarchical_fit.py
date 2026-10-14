"""Hierarchical empirical-Bayes/Laplace group refitting for GBM Toolbox."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import psi

from .individual_fit import individual_fit
from .optimization import Config
from .priors import GaussianPrior, Priors


@dataclass(frozen=True)
class HierarchicalFitConfig:
    """Stopping rule for the hierarchical refitting loop.

    ``tol`` is applied to the change in the group prior between sweeps, so the
    loop ends once the group level has stopped moving rather than after a fixed
    number of passes.
    """

    maxiter: int = 30
    tol: float = 1e-3
    verbose: bool = True
    random_state: int | None = 42

    def __post_init__(self):
        """Validate the stopping rule and the seed."""
        if isinstance(self.maxiter, (bool, np.bool_)) or not isinstance(self.maxiter, (int, np.integer)) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_)) or not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None")


@dataclass
class HierarchicalFitResult:
    """Group-level result: refitted subjects and per-model responsibilities.

    ``responsibilities`` is subjects-by-models and each row sums to one;
    ``model_frequency`` summarises it at the group level.
    """

    models: list
    fits: list
    responsibilities: np.ndarray
    dirichlet_parameters: np.ndarray
    model_frequency: np.ndarray
    group_priors: list[Priors]
    converged: bool
    n_iter: int
    method: str = "hierarchical_empirical_bayes_laplace"
    convergence_history: list[dict] = field(default_factory=list)
    model_seeds: tuple[int, ...] = ()


def _responsibilities(lme, alpha):
    """Posterior probability of each model per subject, from log evidence."""
    log_r = lme + (psi(alpha) - psi(np.sum(alpha)))[None, :]
    log_r -= np.max(log_r, axis=1, keepdims=True)
    r = np.exp(log_r)
    r /= np.sum(r, axis=1, keepdims=True)
    return r


def _max_change(a, b) -> float:
    """Largest absolute elementwise change between two arrays; 0 when empty."""
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _weighted_block_prior(fit, weights, slc, original_prior: GaussianPrior):
    """Responsibility-weighted moment match for one parameter block."""
    means = fit.output.parameters[:, slc]
    d = means.shape[1]
    if d == 0:
        return original_prior
    w = np.asarray(weights, dtype=float)
    sw = float(np.sum(w))
    if sw <= 1e-12:
        return original_prior
    mu = np.sum(w[:, None] * means, axis=0) / sw
    cov = np.zeros((d, d), dtype=float)
    for i in range(len(w)):
        full_cov = fit.math.covariance[i]
        if full_cov is None:
            raise RuntimeError("hierarchical_fit requires valid subject Laplace covariance")
        Si = full_cov[slc, slc]
        delta = means[i] - mu
        cov += w[i] * (Si + np.outer(delta, delta))
    cov /= sw

    # Preserve fixed parameters from the original model exactly.
    fixed = original_prior.fixed_mask
    for j in np.flatnonzero(fixed):
        mu[j] = original_prior.mean[j]
        cov[j, :] = 0.0
        cov[:, j] = 0.0
    free = ~fixed
    if np.any(free):
        block = 0.5 * (cov[np.ix_(free, free)] + cov[np.ix_(free, free)].T)
        vals, vecs = np.linalg.eigh(block)
        floor = max(1e-8, 1e-8 * float(np.max(vals)) if vals.size else 1e-8)
        vals = np.maximum(vals, floor)
        cov[np.ix_(free, free)] = vecs @ np.diag(vals) @ vecs.T
    return GaussianPrior(mu, cov, names=original_prior.names)


def hierarchical_fit(data, models, *, fit_config: Config | None = None, hbi_config: HierarchicalFitConfig | None = None):
    """Hierarchically refit theta/phi group priors across subjects.

    Estimated process/observation-noise priors remain fixed at their model-level
    specification in this first implementation.  Noise parameters are still
    fitted per subject and included in each subject's Hessian/evidence.
    """
    if not isinstance(models, (list, tuple)) or len(models) == 0:
        raise ValueError("models must be a non-empty sequence of StateModel objects")
    if fit_config is None:
        fit_config = Config(verbose=False, display=False, latent_uncertainty="none")
    else:
        fit_config = replace(fit_config, verbose=False, display=False, latent_uncertainty="none")
    if hbi_config is None:
        hbi_config = HierarchicalFitConfig(random_state=fit_config.random_state)
    # Explicit HierarchicalFitConfig seed controls the hierarchical call. Otherwise the
    # seed is inherited from Config. Each model keeps the same stream between
    # sweeps, avoiding unrelated restart draws changing as priors are updated.
    model_seeds = tuple(int(s.generate_state(1)[0]) for s in np.random.SeedSequence(hbi_config.random_state).spawn(len(models)))

    group_priors = [m.priors for m in models]
    alpha = np.ones(len(models), dtype=float)
    prev_freq = alpha / alpha.sum()
    history = []
    converged = False
    fits = None
    r = np.full((len(data), len(models)), 1.0 / len(models))

    for it in range(1, hbi_config.maxiter + 1):
        fits = []
        lme = np.zeros((len(data), len(models)))
        for k, (model, priors) in enumerate(zip(models, group_priors)):
            # replace() changes only theta/phi priors; Q/R modes and noise priors stay untouched.
            mk = replace(model, priors=priors)
            fit = individual_fit(data, mk, config=replace(fit_config, random_state=model_seeds[k]))
            for n, diag in enumerate(fit.math.diagnostics):
                if not diag.laplace_valid:
                    raise RuntimeError(f"hierarchical_fit requires a converged MAP and valid Laplace curvature: model {k}, subject {n}")
            fits.append(fit)
            lme[:, k] = fit.output.log_evidence

        r = _responsibilities(lme, alpha)
        alpha = 1.0 + np.sum(r, axis=0)
        freq = alpha / np.sum(alpha)

        new_priors = []
        for k, (fit, original_model) in enumerate(zip(fits, models)):
            layout = fit.model.parameter_layout
            pe = _weighted_block_prior(fit, r[:, k], layout.theta_slice, original_model.priors.evolution)
            po = _weighted_block_prior(fit, r[:, k], layout.phi_slice, original_model.priors.observation)
            new_priors.append(Priors(pe, po))

        delta_freq = _max_change(freq, prev_freq)
        delta_mean = max(_max_change(p.mean, old.mean) for p, old in zip(new_priors, group_priors))
        delta_cov = max(_max_change(p.covariance, old.covariance) for p, old in zip(new_priors, group_priors))
        history.append({"iteration": it, "frequency_change": delta_freq, "mean_change": delta_mean, "covariance_change": delta_cov})
        if hbi_config.verbose:
            print(f"hierarchical_fit iter {it:02d}: max Δfreq={delta_freq:.3g}, max Δmean={delta_mean:.3g}, max Δcov={delta_cov:.3g}")
        if max(delta_freq, delta_mean, delta_cov) <= hbi_config.tol:
            converged = True
            break
        if it == hbi_config.maxiter:
            break
        group_priors = new_priors
        prev_freq = freq.copy()

    return HierarchicalFitResult(
        models=list(models),
        fits=fits or [],
        responsibilities=r,
        dirichlet_parameters=alpha,
        model_frequency=alpha / np.sum(alpha),
        group_priors=group_priors,
        converged=converged,
        n_iter=it,
        convergence_history=history,
        model_seeds=model_seeds,
    )
