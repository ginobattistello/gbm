"""MAP optimization and observed-Hessian Laplace inference using JAX AD."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
_COMPILED_OBJECTIVE_CACHE = {}

from pathlib import Path

_JAX_CACHE = Path.home() / ".cache" / "gbmtoolbox" / "jax"
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


@dataclass(frozen=True)
class Config:
    """Individual-fit configuration."""

    num_init: int = 4
    random_state: int | None = 42
    verbose: bool = True
    display: bool = False
    maxiter: int = 1000
    hard_bounds: Sequence[tuple[float | None, float | None] | None] | None = None

    hessian_method: str = "autodiff"
    hessian_relative_step: float = 1e-4
    condition_number_warn: float = 1e12

    filter_max_iter: int = 4
    filter_tol: float = 1e-8
    filter_damping: float = 1.0
    filter_jitter: float = 1e-9

    latent_uncertainty: str = "none"
    latent_samples: int | None = 1000
    latent_interval: float | None = 0.95

    def __post_init__(self):
        if not isinstance(self.num_init, int) or self.num_init < 1:
            raise ValueError("num_init must be a positive integer")
        if not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if self.random_state is not None and not isinstance(self.random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer or None")
        if self.hessian_method not in {"autodiff", "central_fd"}:
            raise ValueError("hessian_method must be 'autodiff' or 'central_fd'")
        if self.hessian_relative_step <= 0:
            raise ValueError("hessian_relative_step must be > 0")
        if self.condition_number_warn <= 1:
            raise ValueError("condition_number_warn must be > 1")
        if self.filter_max_iter < 1:
            raise ValueError("filter_max_iter must be >= 1")
        if self.filter_tol <= 0:
            raise ValueError("filter_tol must be > 0")
        if not (0 < self.filter_damping <= 1.0):
            raise ValueError("filter_damping must lie in (0, 1]")
        if self.filter_jitter < 0:
            raise ValueError("filter_jitter must be >= 0")

        mode = str(self.latent_uncertainty).lower()
        if mode not in {"none", "propagated", "filtered"}:
            raise ValueError("latent_uncertainty must be 'none', 'propagated', or 'filtered'")
        object.__setattr__(self, "latent_uncertainty", mode)
        if mode == "propagated":
            if self.latent_samples is None or int(self.latent_samples) < 2:
                raise ValueError("latent_samples must be >= 2 for propagated uncertainty")
            if self.latent_interval is None or not (0 < float(self.latent_interval) < 1):
                raise ValueError("latent_interval must lie in (0, 1) for propagated uncertainty")
            object.__setattr__(self, "latent_samples", int(self.latent_samples))
            object.__setattr__(self, "latent_interval", float(self.latent_interval))
        else:
            object.__setattr__(self, "latent_samples", None)
            object.__setattr__(self, "latent_interval", None)


@dataclass
class StartRecord:
    """Outcome of one L-BFGS-B start, kept so restarts can be compared."""

    initial_parameters: np.ndarray
    final_parameters: np.ndarray
    log_joint: float
    success: bool
    status: int
    message: str
    n_iter: int
    gradient_norm: float


@dataclass
class OptimizationDiagnostics:
    """Everything recorded about a fit other than the estimate itself.

    Covers the multi-start search (``starts``, ``search_path``), the winning
    optimiser run (``lbfgsb_*``, ``abs_grad``) and the separately recomputed
    observed Hessian (``hess_*``, ``laplace_*``). ``laplace_valid`` is false
    when that Hessian is not positive definite, which is reported rather than
    repaired: a good MAP can coexist with an unusable Laplace approximation.
    """

    starts: list[StartRecord] = field(default_factory=list)
    search_path: np.ndarray | None = None
    search_log_joint: np.ndarray | None = None
    lbfgsb_success: bool = False
    lbfgsb_status: int | None = None
    lbfgsb_message: str = ""
    abs_grad: float = np.nan
    n_invalid_evaluations: int = 0
    at_hard_bounds: np.ndarray | None = None
    hess_method: str = "autodiff"
    hess_raw_min_eig: float = np.nan
    hess_raw_max_eig: float = np.nan
    hess_condition_number: float = np.nan
    hess_ill_conditioned: bool = False
    laplace_valid: bool = False
    laplace_fragile: bool = False


@dataclass
class OptimizationResult:
    """MAP estimate for one subject plus its Laplace quantities.

    ``parameters`` is the full vector including any fixed entries, while
    ``free_parameters``, ``hessian`` and ``covariance`` live in the reduced
    free space picked out by ``free_mask``. ``covariance`` and ``log_evidence``
    are ``None``/``nan`` when the observed Hessian is not positive definite.
    """

    parameters: np.ndarray
    free_parameters: np.ndarray
    log_likelihood: float
    log_prior: float
    log_joint: float
    hessian: np.ndarray
    covariance: np.ndarray | None
    log_evidence: float
    diagnostics: OptimizationDiagnostics
    free_mask: np.ndarray


def _free_space(layout):
    """Reduce the prior to the free parameters the optimiser searches."""
    mean = np.asarray(layout.mean, dtype=float)
    covariance = np.asarray(layout.covariance, dtype=float)
    fixed_mask = np.isclose(np.diag(covariance), 0.0, atol=1e-14, rtol=0.0)
    free_mask = ~fixed_mask
    free_mean = mean[free_mask]
    free_covariance = covariance[np.ix_(free_mask, free_mask)]
    if free_covariance.size:
        eig = np.linalg.eigvalsh(free_covariance)
        if np.min(eig) <= 0:
            raise ValueError("free-parameter prior covariance must be positive definite")
        precision = np.linalg.inv(free_covariance)
        sign, logdet = np.linalg.slogdet(free_covariance)
        if sign <= 0:
            raise ValueError("free-parameter prior covariance must be positive definite")
        logdet_covariance = float(logdet)
    else:
        precision = np.zeros((0, 0), dtype=float)
        logdet_covariance = 0.0
    return mean, free_mask, free_mean, free_covariance, precision, logdet_covariance


def _reconstruct_host(free_values, mean, free_mask):
    """Put free-parameter values back into the full vector."""
    full = mean.copy()
    full[free_mask] = np.asarray(free_values, dtype=float)
    return full


def _resolved_hard_bounds(config: Config, model):
    """Return one hard-bound entry per resolved parameter.

    For backward compatibility, a bound vector covering only the historical
    ``[theta, phi]`` blocks is accepted when automatic Q/R parameters are
    present; generated noise parameters are left unbounded.
    """
    if config.hard_bounds is None:
        return None
    bounds = list(config.hard_bounds)
    if len(bounds) == model.priors.dim and model.n_parameters > model.priors.dim:
        bounds.extend([None] * (model.n_parameters - model.priors.dim))
    if len(bounds) != model.n_parameters:
        raise ValueError("hard_bounds must contain one entry per resolved parameter, or one entry per theta/phi parameter when Q/R are estimated internally")
    return tuple(bounds)


def _free_bounds(resolved_bounds, free_mask):
    """Restrict hard bounds to the free parameters, in their order."""
    if resolved_bounds is None:
        return None
    out = []
    for is_free, pair in zip(free_mask, resolved_bounds):
        if not is_free:
            continue
        out.append((None, None) if pair is None else tuple(pair))
    return out


def _clip_to_bounds(x, bounds):
    """Move an initial point inside the L-BFGS-B bounds."""
    x = np.asarray(x, dtype=float).copy()
    if bounds is None:
        return x
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None:
            x[j] = max(x[j], lo + 1e-10)
        if hi is not None:
            x[j] = min(x[j], hi - 1e-10)
    return x


def _at_hard_bounds(parameters, hard_bounds):
    """Identify fitted parameters lying on explicit hard bounds."""
    at_bounds = np.zeros(len(parameters), dtype=bool)
    if hard_bounds is None:
        return at_bounds
    for j, (value, pair) in enumerate(zip(parameters, hard_bounds)):
        if pair is None:
            continue
        lo, hi = pair
        tol = 1e-7 * max(1.0, abs(float(value)))
        if lo is not None and abs(value - lo) <= tol:
            at_bounds[j] = True
        if hi is not None and abs(value - hi) <= tol:
            at_bounds[j] = True
    return at_bounds


def _compiled_objective_functions(model, free_idx, config):
    """Return cached JAX objective/gradient/Hessian functions.

    The compiled functions receive data and prior quantities as arguments,
    rather than capturing their values in Python closures. This allows JAX
    compilation to be reused across repeated fits with the same model
    structure and array shapes.
    """

    free_idx = tuple(int(i) for i in free_idx)
    key = (id(model), free_idx, int(config.filter_max_iter), float(config.filter_tol), float(config.filter_damping), float(config.filter_jitter))
    cached = _COMPILED_OBJECTIVE_CACHE.get(key)
    if cached is not None:
        return cached
    free_idx_j = jnp.asarray(free_idx, dtype=jnp.int32)

    def components(x, full_prior_mean, free_prior_mean, prior_precision, logdet_prior_covariance, prepared_data):
        """Log-likelihood and log-prior at one free-parameter vector."""
        # Reconstruct full parameter vector.
        full = full_prior_mean.at[free_idx_j].set(x)
        run = model.evaluate_jax(
            full,
            prepared_data,
            filter_max_iter=config.filter_max_iter,
            filter_tol=config.filter_tol,
            filter_damping=config.filter_damping,
            filter_jitter=config.filter_jitter,
        )
        loglik = jnp.sum(run["loglik"])
        d = x.shape[0]
        delta = x - free_prior_mean
        quadratic = delta @ prior_precision @ delta
        logprior = -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet_prior_covariance + quadratic)
        return loglik, logprior

    def negative_log_joint(x, full_prior_mean, free_prior_mean, prior_precision, logdet_prior_covariance, prepared_data):
        """Objective minimised by L-BFGS-B: minus the log joint."""
        loglik, logprior = components(x, full_prior_mean, free_prior_mean, prior_precision, logdet_prior_covariance, prepared_data)
        return -(loglik + logprior)

    compiled = {
        "components": jax.jit(components),
        "value": jax.jit(negative_log_joint),
        "value_and_grad": jax.jit(jax.value_and_grad(negative_log_joint)),
        "hessian": jax.jit(jax.hessian(negative_log_joint)),
    }
    _COMPILED_OBJECTIVE_CACHE[key] = compiled
    return compiled


def central_hessian(func, x, relative_step=1e-4):
    """Central finite-difference Hessian retained as a validation backend."""
    x = np.asarray(x, dtype=float).reshape(-1)
    d = x.size
    H = np.zeros((d, d), dtype=float)
    if d == 0:
        return H
    h = relative_step * np.maximum(1.0, np.abs(x))
    f0 = float(func(x))
    if not np.isfinite(f0):
        raise FloatingPointError("objective is non-finite at MAP during Hessian calculation")
    for i in range(d):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        fp, fm = float(func(xp)), float(func(xm))
        H[i, i] = (fp - 2.0 * f0 + fm) / (h[i] ** 2)
        for j in range(i + 1, d):
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm[i] -= h[i]
            xmm[j] -= h[j]
            vals = np.array([func(xpp), func(xpm), func(xmp), func(xmm)], dtype=float)
            if not np.all(np.isfinite(vals)):
                raise FloatingPointError("objective is non-finite near MAP during Hessian calculation")
            Hij = (vals[0] - vals[1] - vals[2] + vals[3]) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = Hij
    return 0.5 * (H + H.T)


def optimize_map(subject_data, model, config: Config, *, rng=None) -> OptimizationResult:
    """Fit one subject: multi-start MAP, then an independent Laplace step.

    The pipeline is deliberately two-stage::

        multi-start L-BFGS-B  ->  MAP  ->  observed Hessian  ->  covariance, evidence

    The quasi-Newton curvature that L-BFGS-B builds up while searching is
    *never* reused for the Laplace approximation. The Hessian is recomputed
    from scratch at the MAP, by automatic differentiation of the same objective
    the optimiser minimised, and it is not repaired by clipping eigenvalues.
    A valid MAP can therefore come back with ``laplace_valid=False``, which is
    reported rather than hidden.

    Parameters fixed by a zero prior variance are removed from the search
    entirely: the optimiser works in the reduced free space and the full vector
    is reassembled afterwards. A model with no free parameters at all is scored
    at its prior mean without optimising.
    """
    from .state_model import prepare_subject_data

    if rng is None:
        rng = np.random.default_rng(config.random_state)
    layout = model.parameter_layout
    mean, free_mask, free_mean, free_covariance, precision, logdet_covariance = _free_space(layout)
    resolved_hard_bounds = _resolved_hard_bounds(config, model)
    bounds = _free_bounds(resolved_hard_bounds, free_mask)
    d = int(np.sum(free_mask))
    free_idx = np.flatnonzero(free_mask)
    prepared = prepare_subject_data(subject_data)

    mean_j = jnp.asarray(mean, dtype=jnp.float64)
    free_mean_j = jnp.asarray(free_mean, dtype=jnp.float64)
    precision_j = jnp.asarray(precision, dtype=jnp.float64)
    logdet_covariance_j = jnp.asarray(logdet_covariance, dtype=jnp.float64)
    compiled = _compiled_objective_functions(model, free_idx, config)

    compiled_components = compiled["components"]
    compiled_value = compiled["value"]
    compiled_value_and_grad = compiled["value_and_grad"]
    compiled_hessian = compiled["hessian"]

    def components_jax(x):
        """Log-likelihood and log-prior at ``x``, with this subject's data bound."""
        return compiled_components(x, mean_j, free_mean_j, precision_j, logdet_covariance_j, prepared)

    def objective_jax(x):
        """Negative log joint at ``x``."""
        return compiled_value(x, mean_j, free_mean_j, precision_j, logdet_covariance_j, prepared)

    def objective_value_and_grad_jax(x):
        """Negative log joint and its exact gradient at ``x``."""
        return compiled_value_and_grad(x, mean_j, free_mean_j, precision_j, logdet_covariance_j, prepared)

    def objective_hessian_jax(x):
        """Observed Hessian of the negative log joint at ``x``."""
        return compiled_hessian(x, mean_j, free_mean_j, precision_j, logdet_covariance_j, prepared)

    if d == 0:
        ll, lp = components_jax(jnp.zeros((0,), dtype=jnp.float64))
        ll = float(ll)
        lp = float(lp)
        diagnostics = OptimizationDiagnostics(
            search_path=np.zeros((1, 0)),
            search_log_joint=np.array([ll + lp]),
            lbfgsb_success=True,
            lbfgsb_status=0,
            lbfgsb_message="all static parameters fixed",
            abs_grad=0.0,
            at_hard_bounds=np.zeros(model.n_parameters, dtype=bool),
            hess_method=config.hessian_method,
            laplace_valid=True,
            laplace_fragile=False,
        )
        return OptimizationResult(mean.copy(), np.zeros(0), ll, lp, ll + lp, np.zeros((0, 0)), np.zeros((0, 0)), ll + lp, diagnostics, free_mask)

    invalid_count = 0

    def scipy_fun(x):
        """Value and gradient for SciPy, counting non-finite evaluations.

        A non-finite objective is reported back as a large finite penalty with
        a zero gradient, so one bad region cannot abort the whole search; the
        occurrences are counted in ``n_invalid_evaluations`` instead.
        """
        nonlocal invalid_count
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        value, grad = objective_value_and_grad_jax(x_jax)
        value_np = float(value)
        grad_np = np.asarray(grad, dtype=float)
        if not np.isfinite(value_np) or not np.all(np.isfinite(grad_np)):
            invalid_count += 1
            return (1e100, np.zeros_like(np.asarray(x, dtype=float)))
        return value_np, grad_np

    init_list = [_clip_to_bounds(free_mean, bounds)]
    for _ in range(config.num_init - 1):
        init_list.append(_clip_to_bounds(rng.multivariate_normal(free_mean, free_covariance), bounds))

    starts = []
    best = None
    best_fval = np.inf
    best_path = None
    best_path_log_joint = None

    for init in init_list:
        path = [np.asarray(init, dtype=float).copy()]

        def callback(xk):
            """Record the search path for this start."""
            # `path` is rebound each iteration and L-BFGS-B calls this
            # synchronously, so the closure always appends to the current start.
            path.append(np.asarray(xk, dtype=float).copy())

        res = minimize(scipy_fun, init, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback, options={"maxiter": config.maxiter})
        fval = float(res.fun)
        if getattr(res, "jac", None) is not None:
            grad = np.asarray(res.jac, dtype=float)
        else:
            _, grad = scipy_fun(res.x)
        logjoint = -fval
        starts.append(
            StartRecord(
                initial_parameters=_reconstruct_host(init, mean, free_mask),
                final_parameters=_reconstruct_host(res.x, mean, free_mask),
                log_joint=logjoint,
                success=bool(res.success and np.isfinite(logjoint)),
                status=int(res.status),
                message=str(res.message),
                n_iter=int(getattr(res, "nit", 0)),
                gradient_norm=float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else np.nan,
            )
        )
        if np.isfinite(fval) and fval < best_fval:
            best = res
            best_fval = float(fval)
            best_path = np.vstack(path)
            if config.display:
                best_path_log_joint = np.asarray([-float(objective_jax(jnp.asarray(z, dtype=jnp.float64))) for z in best_path], dtype=float)
            else:
                best_path_log_joint = None

    if best is None or not np.isfinite(best_fval):
        raise RuntimeError("MAP optimization did not find a finite objective value")

    xhat = np.asarray(best.x, dtype=float)
    full_hat = _reconstruct_host(xhat, mean, free_mask)
    final_ll_j, final_lp_j = components_jax(jnp.asarray(xhat))
    final_ll, final_lp = float(final_ll_j), float(final_lp_j)
    logjoint = final_ll + final_lp
    at_bounds_full = _at_hard_bounds(full_hat, resolved_hard_bounds)
    at_any_bound = bool(np.any(at_bounds_full))

    if config.hessian_method == "autodiff":
        H = np.asarray(objective_hessian_jax(jnp.asarray(xhat, dtype=jnp.float64)), dtype=float)
    else:
        H = central_hessian(lambda z: float(objective_jax(jnp.asarray(z, dtype=jnp.float64))), xhat, config.hessian_relative_step)
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        raise FloatingPointError("observed Hessian contains non-finite values")

    eig = np.linalg.eigvalsh(H)
    min_eig, max_eig = float(np.min(eig)), float(np.max(eig))
    positive_definite = bool(np.all(eig > 0.0))
    condition_number = float(max_eig / min_eig) if positive_definite and min_eig > 0 else np.inf
    ill = bool(positive_definite and (not np.isfinite(condition_number) or condition_number >= config.condition_number_warn))
    laplace_valid = positive_definite
    laplace_fragile = bool(laplace_valid and (ill or at_any_bound))

    if ill and config.verbose:
        warnings.warn("observed Hessian is ill-conditioned; Laplace uncertainty/evidence may be unreliable", RuntimeWarning, stacklevel=2)
    if at_any_bound and config.verbose:
        warnings.warn("MAP estimate lies on a hard bound; the Gaussian Laplace approximation may be unreliable", RuntimeWarning, stacklevel=2)

    if laplace_valid:
        covariance = np.linalg.inv(H)
        sign, logdet_hessian = np.linalg.slogdet(H)
        log_evidence = float(logjoint + 0.5 * d * np.log(2.0 * np.pi) - 0.5 * logdet_hessian) if sign > 0 else np.nan
    else:
        covariance = None
        log_evidence = np.nan

    if getattr(best, "jac", None) is not None:
        best_gradient = np.asarray(best.jac, dtype=float)
    else:
        _, grad_j = objective_value_and_grad_jax(jnp.asarray(xhat, dtype=jnp.float64))
        best_gradient = np.asarray(grad_j, dtype=float)

    diagnostics = OptimizationDiagnostics(
        starts=starts,
        search_path=np.asarray([_reconstruct_host(x, mean, free_mask) for x in best_path]),
        search_log_joint=best_path_log_joint,
        lbfgsb_success=bool(best.success),
        lbfgsb_status=int(best.status),
        lbfgsb_message=str(best.message),
        abs_grad=float(np.linalg.norm(best_gradient)) if np.all(np.isfinite(best_gradient)) else np.nan,
        n_invalid_evaluations=invalid_count,
        at_hard_bounds=at_bounds_full,
        hess_method=config.hessian_method,
        hess_raw_min_eig=min_eig,
        hess_raw_max_eig=max_eig,
        hess_condition_number=condition_number,
        hess_ill_conditioned=ill,
        laplace_valid=laplace_valid,
        laplace_fragile=laplace_fragile,
    )
    return OptimizationResult(full_hat, xhat, final_ll, final_lp, logjoint, H, covariance, log_evidence, diagnostics, free_mask)
