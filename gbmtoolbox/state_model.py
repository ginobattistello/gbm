"""Generative state-model definition and JAX likelihood helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .families import get_family
from .parameters import ParameterLayout, build_parameter_layout
from .priors import GaussianPrior, Priors, covariance_matrix

Family = str


def trial_input(u: Any, t: int, T: int):
    """Host-side extraction of modeller-defined inputs for trial ``t``."""
    if u is None:
        return None
    if isinstance(u, dict):
        out = {}
        for key, value in u.items():
            arr = np.asarray(value)
            if arr.dtype.kind not in "biufc":
                raise TypeError(f"input field {key!r} must be numeric/bool for the JAX backend")
            if arr.ndim == 0:
                out[key] = value
            else:
                if arr.shape[0] != T:
                    raise ValueError(f"input field {key!r} has first dimension {arr.shape[0]}, expected {T}")
                out[key] = value[t]
        return out
    arr = np.asarray(u)
    if arr.dtype.kind not in "biufc":
        raise TypeError("trial input must be numeric/bool for the JAX backend")
    if arr.ndim == 0:
        return u
    if arr.shape[0] != T:
        raise ValueError(f"trial input has first dimension {arr.shape[0]}, expected {T}")
    return u[t]


def _prepare_u(u: Any, T: int):
    if u is None:
        return None
    if isinstance(u, dict):
        out = {}
        for key, value in u.items():
            arr = np.asarray(value)
            if arr.dtype.kind not in "biufc":
                raise TypeError(f"input field {key!r} must be numeric/bool for the JAX backend")
            if arr.ndim == 0:
                arr = np.broadcast_to(arr, (T,))
            elif arr.shape[0] != T:
                raise ValueError(f"input field {key!r} has first dimension {arr.shape[0]}, expected {T}")
            out[key] = jnp.asarray(arr)
        return out
    arr = np.asarray(u)
    if arr.dtype.kind not in "biufc":
        raise TypeError("trial input must be numeric/bool for the JAX backend")
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, (T,))
    elif arr.shape[0] != T:
        raise ValueError(f"trial input has first dimension {arr.shape[0]}, expected {T}")
    return jnp.asarray(arr)


def prepare_subject_data(subject_data: dict):
    """Convert one validated subject to a fixed-shape JAX pytree."""
    y = np.asarray(subject_data["y"])
    if y.ndim == 0 or y.shape[0] == 0:
        raise ValueError("y must contain at least one trial")
    T = int(y.shape[0])
    return {"y": jnp.asarray(y), "u": _prepare_u(subject_data.get("u", None), T)}


def _expand_covariance_jax(value, dim: int):
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim == 0:
        return jnp.eye(dim, dtype=arr.dtype) * arr
    if arr.ndim == 1:
        return jnp.diag(arr)
    return arr


def _mode(value):
    if value is None:
        return "none"
    if isinstance(value, str):
        v = value.lower()
        if v not in {"none", "diagonal"}:
            raise ValueError("covariance string mode must be 'none' or 'diagonal'")
        return v
    return "custom" if callable(value) else "fixed"


@dataclass(frozen=True)
class StateModel:
    """Generative model used by all GBM Toolbox inference routines.

    Trial order is ``x_t -> predictor eta_t -> likelihood(y_t) -> evolution -> x_{t+1}``.

    For Gaussian outcomes ``observation`` returns the mean.  For Bernoulli and
    categorical outcomes it returns logits (not probabilities).
    """

    evolution: Callable
    observation: Callable
    family: Family
    priors: Priors
    initial_state: Sequence[float]
    initial_state_covariance: object | None = None
    process_covariance: object | None = None
    observation_covariance: object | None = None
    process_noise_prior: GaussianPrior | None = None
    observation_noise_prior: GaussianPrior | None = None
    observation_dim: int | None = None
    state_names: Sequence[str] | None = None
    name: str | None = None

    def __post_init__(self):
        family = str(self.family).lower()
        get_family(family)
        x0 = np.asarray(self.initial_state, dtype=float).reshape(-1)
        if x0.size == 0 or not np.all(np.isfinite(x0)):
            raise ValueError("initial_state must be a non-empty finite vector")
        if self.state_names is None:
            state_names = tuple(f"x[{i}]" for i in range(x0.size))
        else:
            state_names = tuple(str(x) for x in self.state_names)
            if len(state_names) != x0.size:
                raise ValueError("state_names must contain one name per latent-state dimension")
            if len(set(state_names)) != len(state_names):
                raise ValueError("state_names must be unique")

        qmode = _mode(self.process_covariance)
        rmode = _mode(self.observation_covariance)
        if family != "gaussian" and rmode != "none":
            raise ValueError("observation_covariance is only used for Gaussian outcomes")
        if qmode == "diagonal" and self.process_noise_prior is None:
            raise ValueError("process_noise_prior is required when process_covariance='diagonal'")
        if self.observation_dim is not None and int(self.observation_dim) < 1:
            raise ValueError("observation_dim must be >= 1")

        object.__setattr__(self, "family", family)
        object.__setattr__(self, "initial_state", x0)
        object.__setattr__(self, "state_names", state_names)
        object.__setattr__(self, "observation_dim", None if self.observation_dim is None else int(self.observation_dim))
        object.__setattr__(self, "name", self.name or "StateModel")

    @property
    def n_state(self) -> int:
        return int(self.initial_state.size)

    @property
    def n_theta(self) -> int:
        return self.priors.evolution.dim

    @property
    def n_phi(self) -> int:
        return self.priors.observation.dim

    @property
    def process_covariance_mode(self) -> str:
        return _mode(self.process_covariance)

    @property
    def observation_covariance_mode(self) -> str:
        mode = _mode(self.observation_covariance)
        if mode == "none" and self.family == "gaussian":
            # Gaussian outcomes have no likelihood without R, so an omitted
            # observation_covariance means "estimate it" rather than "no noise".
            return "diagonal"
        return mode

    @property
    def resolved_observation_dim(self) -> int:
        if self.observation_covariance_mode != "diagonal":
            return 0
        if self.observation_dim is not None:
            return self.observation_dim
        if self.observation_noise_prior is not None and self.observation_noise_prior.dim > 1:
            return self.observation_noise_prior.dim
        return 1

    @cached_property
    def parameter_layout(self) -> ParameterLayout:
        return build_parameter_layout(
            self.priors,
            process_noise_dim=self.n_state if self.process_covariance_mode == "diagonal" else 0,
            process_noise_prior=self.process_noise_prior,
            observation_noise_dim=self.resolved_observation_dim if self.observation_covariance_mode == "diagonal" else 0,
            observation_noise_prior=self.observation_noise_prior,
        )

    @property
    def n_parameters(self) -> int:
        return self.parameter_layout.dim

    def split_parameters(self, parameters):
        """Backward-compatible split returning only ``theta, phi``."""
        p = parameters
        return p[self.parameter_layout.theta_slice], p[self.parameter_layout.phi_slice]

    def unpack_parameters(self, parameters):
        return self.parameter_layout.unpack(parameters)

    def initial_covariance_matrix(self) -> np.ndarray:
        if self.initial_state_covariance is None:
            return np.zeros((self.n_state, self.n_state), dtype=float)
        return covariance_matrix(self.initial_state_covariance, self.n_state, name="initial_state_covariance", allow_semidefinite=True)

    def initial_covariance_jax(self):
        return jnp.asarray(self.initial_covariance_matrix(), dtype=jnp.float64)

    def process_covariance_jax(self, theta, rho_q, u_t):
        mode = self.process_covariance_mode
        if mode == "none":
            return jnp.zeros((self.n_state, self.n_state), dtype=jnp.float64)
        if mode == "diagonal":
            return jnp.diag(jnp.exp(2.0 * rho_q))
        value = self.process_covariance(theta, u_t) if mode == "custom" else self.process_covariance
        return _expand_covariance_jax(value, self.n_state)

    def observation_covariance_jax(self, phi, rho_r, u_t, dim: int):
        mode = self.observation_covariance_mode
        if mode == "diagonal":
            return jnp.diag(jnp.exp(2.0 * rho_r))
        if mode == "none":
            raise ValueError("Gaussian observation covariance cannot be None")
        value = self.observation_covariance(phi, u_t) if mode == "custom" else self.observation_covariance
        return _expand_covariance_jax(value, dim)

    def process_covariance_matrix(self, theta, u_t, rho_q=None) -> np.ndarray:
        rho_q = jnp.zeros((0,)) if rho_q is None else jnp.asarray(rho_q)
        return np.asarray(self.process_covariance_jax(jnp.asarray(theta), rho_q, u_t), dtype=float)

    def observation_covariance_matrix(self, phi, u_t, dim: int, rho_r=None) -> np.ndarray:
        rho_r = jnp.zeros((0,)) if rho_r is None else jnp.asarray(rho_r)
        return np.asarray(self.observation_covariance_jax(jnp.asarray(phi), rho_r, u_t, dim), dtype=float)

    def has_state_uncertainty(self) -> bool:
        if np.any(self.initial_covariance_matrix() != 0.0):
            return True
        mode = self.process_covariance_mode
        if mode == "none":
            return False
        if mode in {"diagonal", "custom"}:
            return True
        return bool(np.any(np.asarray(self.process_covariance, dtype=float) != 0.0))

    def deterministic_run_jax(self, parameters, prepared_data):
        theta, phi, _, rho_r = self.unpack_parameters(parameters)
        family = get_family(self.family)
        y = prepared_data["y"]
        u = prepared_data["u"]

        def one_step(x, y_t, u_t):
            eta = jnp.atleast_1d(self.observation(x, phi, u_t))
            R = None
            if self.family == "gaussian":
                R = self.observation_covariance_jax(phi, rho_r, u_t, eta.shape[0])
            ll = family.log_prob(y_t, eta, R)
            pred = family.mean(eta, R)
            x_next = jnp.asarray(self.evolution(x, theta, u_t, y_t), dtype=jnp.float64).reshape(x.shape)
            return x_next, (x, pred, ll)

        if u is None:

            def step(x, y_t):
                return one_step(x, y_t, None)

            _, (states, predictions, loglik) = jax.lax.scan(step, jnp.asarray(self.initial_state), y)
        else:

            def step(x, inp):
                y_t, u_t = inp
                return one_step(x, y_t, u_t)

            _, (states, predictions, loglik) = jax.lax.scan(step, jnp.asarray(self.initial_state), (y, u))

        return {
            "loglik": loglik,
            "states": states,
            "state_covariance": jnp.zeros((y.shape[0], self.n_state, self.n_state), dtype=jnp.float64),
            "prediction": predictions,
        }

    def evaluate_jax(
        self, parameters, prepared_data, *, filter_max_iter: int = 8, filter_tol: float = 1e-8, filter_damping: float = 1.0, filter_jitter: float = 1e-9
    ):
        if self.has_state_uncertainty():
            from .filtering import nonlinear_state_filter_jax

            return nonlinear_state_filter_jax(
                self, parameters, prepared_data, max_update_iter=filter_max_iter, update_tol=filter_tol, damping=filter_damping, jitter=filter_jitter
            )
        return self.deterministic_run_jax(parameters, prepared_data)

    def evaluate(self, parameters, subject_data, *, config=None):
        prepared = prepare_subject_data(subject_data)
        kwargs = {}
        if config is not None:
            kwargs = {
                "filter_max_iter": config.filter_max_iter,
                "filter_tol": config.filter_tol,
                "filter_damping": config.filter_damping,
                "filter_jitter": config.filter_jitter,
            }
        out = self.evaluate_jax(jnp.asarray(parameters, dtype=jnp.float64), prepared, **kwargs)
        host = jax.tree_util.tree_map(lambda z: np.asarray(z), out)
        host["filtering_method"] = "generalized_gaussian_fisher" if self.has_state_uncertainty() else "deterministic"
        pred = host["prediction"]
        if self.family in {"gaussian", "bernoulli"} and pred.ndim == 2 and pred.shape[1] == 1:
            host["prediction"] = pred[:, 0]
        return host

    def __call__(self, parameters, subject_data):
        return self.evaluate(parameters, subject_data)["loglik"]
