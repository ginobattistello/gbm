"""Central preflight validation for GBM Toolbox JAX model specifications."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .priors import covariance_matrix
from .state_model import prepare_subject_data, trial_input


@dataclass(frozen=True)
class ValidatedSpec:
    n_subjects: int
    n_parameters: int
    n_state: int
    parameter_names: tuple[str, ...]
    state_names: tuple[str, ...]


def _check_numeric_finite(value, name: str):
    arr = np.asarray(value)
    if arr.dtype.kind not in "biufc":
        raise TypeError(f"{name} must be numeric/bool for the JAX backend")
    if not np.all(np.isfinite(arr.astype(float))):
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_u(u, T: int, subject: int):
    if u is None:
        return
    if isinstance(u, dict):
        for key, value in u.items():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] != T:
                raise ValueError(f"subject {subject}: input field {key!r} has length {arr.shape[0]}, expected {T}")
            _check_numeric_finite(value, f"subject {subject} input {key!r}")
        return
    arr = np.asarray(u)
    if arr.ndim > 0 and arr.shape[0] != T:
        raise ValueError(f"subject {subject}: u has first dimension {arr.shape[0]}, expected {T}")
    _check_numeric_finite(u, f"subject {subject} u")


def validate_fit_spec(data, model, config) -> ValidatedSpec:
    """Check modeller inputs and JAX traceability before optimization."""
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError("data must be a non-empty list/tuple of subject dictionaries")

    layout = model.parameter_layout
    parameter_names = tuple(layout.names)
    if len(parameter_names) != model.n_parameters:
        raise ValueError("parameter names are inconsistent with resolved parameter dimension")
    if len(model.state_names) != model.n_state:
        raise ValueError("state_names are inconsistent with initial_state")

    if config.hard_bounds is not None:
        n_bounds = len(config.hard_bounds)
        valid_lengths = {model.n_parameters}
        if model.n_parameters > model.priors.dim:
            valid_lengths.add(model.priors.dim)
        if n_bounds not in valid_lengths:
            raise ValueError(
                "hard_bounds must contain one entry per resolved parameter, or one entry per theta/phi parameter when Q/R are estimated internally"
            )
        for j, pair in enumerate(config.hard_bounds):
            if pair is None:
                continue
            if len(pair) != 2:
                raise ValueError(f"hard_bounds[{j}] must be (low, high) or None")
            lo, hi = pair
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError(f"hard_bounds[{j}] must satisfy low < high")

    theta0, phi0, rho_q0, rho_r0 = layout.unpack(jnp.asarray(layout.mean))

    for n, subject in enumerate(data):
        if not isinstance(subject, dict) or "y" not in subject:
            raise ValueError(f"subject {n}: each subject must be a dict containing 'y'")
        if "u" not in subject:
            raise ValueError(f"subject {n}: each subject must contain the generic input field 'u'")
        y = np.asarray(subject["y"])
        if y.ndim == 0 or y.shape[0] == 0:
            raise ValueError(f"subject {n}: y must contain at least one trial")
        _check_numeric_finite(y, f"subject {n} y")
        T = y.shape[0]
        _validate_u(subject.get("u"), T, n)

        if model.family == "bernoulli":
            yf = np.asarray(y, dtype=float).reshape(-1)
            if not np.all(np.isin(yf, [0.0, 1.0])):
                raise ValueError(f"subject {n}: Bernoulli y must contain only 0 and 1")
        elif model.family == "categorical":
            yf = np.asarray(y, dtype=float).reshape(-1)
            if not np.all(np.equal(yf, np.floor(yf))) or np.any(yf < 0):
                raise ValueError(f"subject {n}: categorical y must contain non-negative integer labels")
        else:
            np.asarray(y, dtype=float)

        x0 = jnp.asarray(model.initial_state)
        u0_host = trial_input(subject.get("u"), 0, T)
        u0 = jax.tree_util.tree_map(jnp.asarray, u0_host) if u0_host is not None else None
        y0 = jnp.asarray(y[0])

        try:
            eta1 = jnp.atleast_1d(model.observation(x0, phi0, u0))
            eta2 = jnp.atleast_1d(model.observation(x0, phi0, u0))
            eta1_np, eta2_np = np.asarray(eta1, dtype=float), np.asarray(eta2, dtype=float)
        except Exception as exc:
            raise ValueError(f"subject {n}: observation must be JAX-compatible and return numeric mean/logits; original error: {exc}") from exc
        if eta1_np.shape != eta2_np.shape or not np.allclose(eta1_np, eta2_np):
            raise ValueError("observation must be deterministic for fixed inputs")
        if eta1_np.ndim != 1 or eta1_np.size == 0 or not np.all(np.isfinite(eta1_np)):
            raise ValueError(f"subject {n}: observation must return a finite scalar/vector")

        if model.family == "bernoulli" and eta1_np.size != 1:
            raise ValueError("Bernoulli observation must return one logit")
        if model.family == "categorical":
            if eta1_np.size < 2:
                raise ValueError("categorical observation must return at least two logits")
            if np.max(np.asarray(y, dtype=int)) >= eta1_np.size:
                raise ValueError(f"subject {n}: categorical outcome exceeds number of returned logits")
        if model.family == "gaussian":
            ydim = np.asarray(y[0], dtype=float).reshape(-1).size
            if eta1_np.size != ydim:
                raise ValueError(f"subject {n}: Gaussian outcome dimension {ydim} differs from observation mean dimension {eta1_np.size}")
            if model.observation_covariance_mode == "diagonal" and model.resolved_observation_dim != eta1_np.size:
                raise ValueError(
                    f"subject {n}: estimated diagonal R has dimension {model.resolved_observation_dim}, "
                    f"but the Gaussian observation returns {eta1_np.size} values. "
                    f"For multivariate Gaussian outcomes set observation_dim={eta1_np.size}, "
                    "or provide a matching vector observation_noise_prior."
                )
            R = np.asarray(model.observation_covariance_jax(phi0, rho_r0, u0, eta1_np.size), dtype=float)
            covariance_matrix(R, eta1_np.size, name="observation_covariance", allow_semidefinite=False)

        try:
            x_next = jnp.asarray(model.evolution(x0, theta0, u0, y0)).reshape(-1)
            x_next_np = np.asarray(x_next, dtype=float)
            x_repeat_np = np.asarray(jnp.asarray(model.evolution(x0, theta0, u0, y0)).reshape(-1), dtype=float)
        except Exception as exc:
            raise ValueError(f"subject {n}: evolution must be JAX-compatible; original error: {exc}") from exc
        if x_next_np.shape != (model.n_state,) or not np.all(np.isfinite(x_next_np)):
            raise ValueError("evolution must return a finite state with the same shape as initial_state")
        # A host-side RNG inside evolution would be traced once and silently frozen
        # into the compiled graph, so nondeterminism must be caught before tracing.
        if x_next_np.shape != x_repeat_np.shape or not np.allclose(x_next_np, x_repeat_np):
            raise ValueError("evolution must be deterministic for fixed inputs")

        if model.process_covariance_mode != "none":
            Q = np.asarray(model.process_covariance_jax(theta0, rho_q0, u0), dtype=float)
            covariance_matrix(Q, model.n_state, name="process_covariance", allow_semidefinite=True)

    # Filtered uncertainty is only meaningful when the state model is stochastic.
    # A deterministic model yields an identically zero state covariance, which
    # would otherwise be reported as genuine filtered uncertainty.
    if config.latent_uncertainty == "filtered" and not model.has_state_uncertainty():
        raise ValueError(
            "latent_uncertainty='filtered' requires initial-state uncertainty or process_covariance; "
            "use 'propagated' or 'none' for a deterministic state model"
        )

    # Explicit JAX smoke test on one complete subject likelihood and its gradient.
    prepared = prepare_subject_data(data[0])
    p0 = jnp.asarray(layout.mean, dtype=jnp.float64)

    def objective(p):
        run = model.evaluate_jax(
            p,
            prepared,
            filter_max_iter=config.filter_max_iter,
            filter_tol=config.filter_tol,
            filter_damping=config.filter_damping,
            filter_jitter=config.filter_jitter,
        )
        return -jnp.sum(run["loglik"])

    try:
        value = jax.jit(objective)(p0)
        grad = jax.jit(jax.grad(objective))(p0)
        if not np.isfinite(float(value)) or not np.all(np.isfinite(np.asarray(grad))):
            raise ValueError("JAX likelihood or gradient is non-finite at the prior mean")
    except Exception as exc:
        raise ValueError(
            "model failed the JAX trace/gradient preflight. Use jax.numpy operations, avoid Python int/float casts of traced values, "
            "in-place mutation, data-dependent Python branching, and NumPy inside evolution/observation/covariance callables. "
            f"Original error: {exc}"
        ) from exc

    return ValidatedSpec(len(data), model.n_parameters, model.n_state, parameter_names, tuple(model.state_names))
