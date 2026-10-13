"""Individual MAP/Laplace fitting for GBM Toolbox StateModel objects."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .optimization import Config, OptimizationDiagnostics, OptimizationResult, optimize_map
from .priors import GammaPrecisionPrior
from .state_model import filter_settings, prepare_subject_data
from .validation import validate_fit_spec

_PROPAGATION_CACHE = {}


def _noise_prior_description(prior):
    """Record the resolved distribution rather than only its moment proxy."""
    if prior is None:
        return None
    if isinstance(prior, GammaPrecisionPrior):
        return {
            "distribution": "gamma_precision",
            "shape": prior.shape.copy(),
            "rate": prior.rate.copy(),
            "parameterization": "log_standard_deviation",
            "names": tuple(prior.names or ()),
        }
    return {
        "distribution": "gaussian_log_sd",
        "mean": prior.mean.copy(),
        "covariance": prior.covariance.copy(),
        "parameterization": "log_standard_deviation",
        "names": tuple(prior.names or ()),
    }


def _compiled_propagation(model, config):
    """Cache the JAX function that reruns the model at posterior draws."""
    key = (model.execution_key, filter_settings(config))

    cached = _PROPAGATION_CACHE.get(key)
    if cached is not None:
        return cached

    def one_draw(parameters, prepared_data):
        """Re-run the model at one posterior draw of the static parameters."""
        run = model.evaluate_jax(
            parameters,
            prepared_data,
            filter_max_iter=config.filter_max_iter,
            filter_tol=config.filter_tol,
            filter_damping=config.filter_damping,
            filter_jitter=config.filter_jitter,
        )
        return run["states"]

    batched = jax.jit(jax.vmap(one_draw, in_axes=(0, None)))

    _PROPAGATION_CACHE[key] = batched
    return batched


@dataclass
class FitInput:
    """What was fitted: model identity, parameter layout and the prior.

    Recorded so a result can be interpreted, and re-checked, without the model
    object that produced it.
    """

    model_name: str
    family: str
    parameter_names: tuple[str, ...]
    state_names: tuple[str, ...]
    theta_slice: slice
    phi_slice: slice
    process_noise_slice: slice
    observation_noise_slice: slice
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    process_noise_prior: dict | None = None
    observation_noise_prior: dict | None = None


@dataclass
class FitOutput:
    """The estimates, one row per subject.

    ``parameters`` is the full fitted vector; the ``*_parameters`` fields are
    views of its blocks, and the ``*_sd`` fields are the noise parameters
    exponentiated back to standard deviations. ``prediction`` holds trial-wise
    expected outcomes -- probabilities for Bernoulli and categorical families,
    not logits.
    """

    parameters: np.ndarray
    evolution_parameters: np.ndarray
    observation_parameters: np.ndarray
    process_noise_parameters: np.ndarray
    observation_noise_parameters: np.ndarray
    process_noise_sd: np.ndarray
    observation_noise_sd: np.ndarray
    log_evidence: np.ndarray
    latent: list[dict]
    prediction: list[np.ndarray]


@dataclass
class FitMath:
    """Quantities behind the estimates: densities, curvature and diagnostics.

    ``hessian`` lives in each subject's free space, selected by ``free_mask``.
    ``covariance`` is embedded back into the full parameter vector, with zero
    rows and columns for fixed parameters, and is ``None`` where the Laplace
    approximation was rejected.
    """

    log_likelihood: np.ndarray
    log_prior: np.ndarray
    log_joint: np.ndarray
    hessian: list[np.ndarray]
    covariance: list[np.ndarray | None]
    diagnostics: list[OptimizationDiagnostics]
    free_mask: list[np.ndarray]


@dataclass
class FitProfile:
    """When the fit ran and under which configuration."""

    datetime: str
    config: Config
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class FitResult:
    """Everything one call to :func:`individual_fit` produced.

    Split four ways so each part can be read on its own: ``input`` is what was
    asked for, ``output`` the estimates, ``math`` the quantities behind them,
    and ``profile`` the run's provenance. The model and data are kept too, so
    diagnostics can re-evaluate the fit without them being passed around
    separately.
    """

    input: FitInput
    output: FitOutput
    math: FitMath
    profile: FitProfile
    model: Any
    data: Any
    method: str = "MAP/Laplace"

    def plot(self, subject: int = 0, **kwargs):
        """Plot one subject's fit; see :func:`gbmtoolbox.display.plot_subject`."""
        from .display import plot_subject

        return plot_subject(self, subject=subject, **kwargs)

    def summary(self, subject: int = 0) -> str:
        """One-screen text summary of one subject's fit."""
        from .reporting import fit_summary

        return fit_summary(self, subject=subject)

    def __repr__(self):
        """Show the first subject's summary rather than the raw dataclass."""
        return self.summary(0)


def _full_covariance(opt: OptimizationResult, n_params: int) -> np.ndarray | None:
    """Embed a free-space covariance back into the full parameter vector."""
    if opt.covariance is None:
        return None
    full = np.zeros((n_params, n_params), dtype=float)
    idx = np.flatnonzero(opt.free_mask)
    full[np.ix_(idx, idx)] = opt.covariance
    return full


def _latent_none(run, model):
    """Latent states with no uncertainty attached."""
    mean = np.asarray(run["states"], dtype=float)
    return {
        "state": {
            "mean": mean,
            "covariance": None,
            "sd": None,
            "interval_low": None,
            "interval_high": None,
            "interval_mass": None,
            "interval_method": None,
            "uncertainty_type": "none",
            "method": run.get("filtering_method", "deterministic"),
            "state_names": tuple(model.state_names),
        }
    }


def _latent_filtered(run, model):
    """Latent states with the filter's own conditional uncertainty."""
    mean = np.asarray(run["states"], dtype=float)
    cov = np.asarray(run["state_covariance"], dtype=float)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    return {
        "state": {
            "mean": mean,
            "covariance": cov,
            "sd": sd,
            "interval_low": None,
            "interval_high": None,
            "interval_mass": None,
            "interval_method": None,
            "uncertainty_type": "filtered",
            "method": run.get("filtering_method", "generalized_gaussian_fisher"),
            "state_names": tuple(model.state_names),
        }
    }


def _sample_static_posterior(opt: OptimizationResult, model, n_samples: int, rng):
    """Draw static parameters from the Laplace posterior."""
    if not opt.diagnostics.laplace_valid or opt.covariance is None:
        raise ValueError("propagated latent uncertainty requires a valid Laplace posterior")
    free_draws = rng.multivariate_normal(opt.free_parameters, opt.covariance, size=n_samples)
    draws = np.tile(model.parameter_layout.mean, (n_samples, 1))
    draws[:, opt.free_mask] = free_draws
    return draws


def _latent_propagated(opt, model, subject_data, config, rng, *, prepared_data=None, run_map=None):
    """Latent uncertainty from rerunning the model at posterior draws."""
    if not opt.diagnostics.laplace_valid:
        warnings.warn(
            "Laplace posterior is invalid; propagated latent uncertainty is unavailable. The MAP latent trajectory is retained without a shadow.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _latent_none(run_map if run_map is not None else model.evaluate(opt.parameters, subject_data, config=config, prepared_data=prepared_data), model)
    if opt.diagnostics.laplace_fragile:
        warnings.warn("Laplace posterior is numerically fragile; propagated latent uncertainty may be sensitive.", RuntimeWarning, stacklevel=3)
    draws = _sample_static_posterior(opt, model, config.latent_samples, rng)
    prepared = prepare_subject_data(subject_data) if prepared_data is None else prepared_data
    propagate = _compiled_propagation(model, config)
    samples = np.asarray(propagate(jnp.asarray(draws, dtype=jnp.float64), prepared), dtype=float)
    mean = np.mean(samples, axis=0)
    centered = samples - mean[None, :, :]
    cov = np.einsum("sti,stj->tij", centered, centered) / (samples.shape[0] - 1)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    alpha = (1.0 - config.latent_interval) / 2.0
    low = np.quantile(samples, alpha, axis=0)
    high = np.quantile(samples, 1.0 - alpha, axis=0)
    return {
        "state": {
            "mean": mean,
            "covariance": cov,
            "sd": sd,
            "interval_low": low,
            "interval_high": high,
            "interval_mass": config.latent_interval,
            "interval_method": "empirical_quantile",
            "uncertainty_type": "propagated",
            "method": "laplace_posterior_sampling",
            "state_names": tuple(model.state_names),
        }
    }


def _subject_list(data, u, family: str) -> list:
    """Normalise ``individual_fit`` input to a list of ``{"y", "u"}`` subject dicts.

    A list or tuple containing dicts is the multi-subject form and is returned
    unchanged. Anything else is one subject's outcomes: a vector with one entry
    per trial (or, for a multivariate Gaussian, a ``(trials, outputs)`` array),
    paired with ``u``. Stacked subjects are rejected rather than guessed at,
    because a 2-D array cannot say whether its rows are subjects or trials.
    """
    if isinstance(data, (list, tuple)) and any(isinstance(item, dict) for item in data):
        if u is not None:
            raise ValueError("u is only for a single subject's outcome vector; with a list of subjects, put each subject's inputs in its own dict")
        return list(data)
    if isinstance(data, (list, tuple)) and len(data) == 0:
        return list(data)
    y = np.asarray(data)
    if isinstance(data, (list, tuple)) and y.ndim != 1:
        raise ValueError("pass several subjects as a list of {'y': ..., 'u': ...} dicts, or one subject's outcomes as a single vector")
    allowed = (1, 2) if family == "gaussian" else (1,)
    if y.ndim not in allowed:
        shape = "a vector (trials,) or a (trials, outputs) array" if family == "gaussian" else "a vector with one entry per trial"
        raise ValueError(f"a single subject's {family} outcomes must be {shape}, got shape {y.shape}; fit several subjects as a list of dicts")
    return [{"y": y, "u": u}]


def individual_fit(data, model, *, u=None, config: Config | None = None) -> FitResult:
    """Fit every subject under one :class:`~gbmtoolbox.StateModel`.

    ``data`` is either a list of subject dicts, ``[{"y": ..., "u": ...}, ...]``,
    or a single subject's outcome vector ``y`` with one entry per trial. In the
    single-subject form, trial inputs are passed as ``u``: an array of any
    dimension whose first axis is trials (``u[t]`` is what the model receives on
    trial ``t``), a dict of such arrays, or ``None``::

        fit = individual_fit(y, model, u=u)  # y: (T,), u: (T, ...)

    Both forms run the same fit; the result stores the subject as a dict, so
    ``fit.data[0]["y"]`` works either way.
    """
    fit_started = perf_counter()
    if config is None:
        config = Config()
    data = _subject_list(data, u, model.family)
    validated = validate_fit_spec(data, model, config)
    # Stable, separate streams: uncertainty draws must not change MAP starts
    # for subsequent subjects. Adding subjects at the end preserves earlier ones.
    subject_seeds = np.random.SeedSequence(config.random_state).spawn(len(data))
    opts = []
    latent = []
    predictions = []

    for n, subject in enumerate(data):
        subject_started = perf_counter()
        prepared = validated.prepared_subjects[n]
        fit_seed, latent_seed = subject_seeds[n].spawn(2)
        fit_rng = np.random.default_rng(fit_seed)
        latent_rng = np.random.default_rng(latent_seed)
        opt = optimize_map(subject, model, config, rng=fit_rng, prepared_data=prepared, objective_context=validated.objective_context if n == 0 else None)
        opts.append(opt)
        prediction_started = perf_counter()
        run_map = model.evaluate(opt.parameters, subject, config=config, prepared_data=prepared)
        predictions.append(np.asarray(run_map["prediction"], dtype=float))
        opt.diagnostics.timings["prediction"] = perf_counter() - prediction_started
        latent_started = perf_counter()
        if config.latent_uncertainty == "propagated":
            latent_n = _latent_propagated(opt, model, subject, config, latent_rng, prepared_data=prepared, run_map=run_map)
        elif config.latent_uncertainty == "filtered":
            latent_n = _latent_filtered(run_map, model)
        else:
            latent_n = _latent_none(run_map, model)
        latent.append(latent_n)
        opt.diagnostics.timings["latent_uncertainty"] = perf_counter() - latent_started
        opt.diagnostics.timings["subject_total"] = perf_counter() - subject_started
        if config.verbose:
            status = "valid" if opt.diagnostics.laplace_valid else "INVALID"
            frag = " (fragile)" if opt.diagnostics.laplace_fragile else ""
            print(f"Subject {n + 1:02d}: log joint={opt.log_joint:.3f}, Laplace={status}{frag}")

    parameters = np.vstack([o.parameters for o in opts])
    layout = model.parameter_layout
    pq = parameters[:, layout.process_noise_slice]
    pr = parameters[:, layout.observation_noise_slice]

    result = FitResult(
        input=FitInput(
            model_name=model.name,
            family=model.family,
            parameter_names=tuple(layout.names),
            state_names=tuple(model.state_names),
            theta_slice=layout.theta_slice,
            phi_slice=layout.phi_slice,
            process_noise_slice=layout.process_noise_slice,
            observation_noise_slice=layout.observation_noise_slice,
            prior_mean=layout.mean.copy(),
            prior_covariance=layout.covariance.copy(),
            process_noise_prior=_noise_prior_description(layout.process_noise_prior),
            observation_noise_prior=_noise_prior_description(layout.observation_noise_prior),
        ),
        output=FitOutput(
            parameters=parameters,
            evolution_parameters=parameters[:, layout.theta_slice],
            observation_parameters=parameters[:, layout.phi_slice],
            process_noise_parameters=pq,
            observation_noise_parameters=pr,
            process_noise_sd=np.exp(pq),
            observation_noise_sd=np.exp(pr),
            log_evidence=np.asarray([o.log_evidence for o in opts], dtype=float),
            latent=latent,
            prediction=predictions,
        ),
        math=FitMath(
            log_likelihood=np.asarray([o.log_likelihood for o in opts], dtype=float),
            log_prior=np.asarray([o.log_prior for o in opts], dtype=float),
            log_joint=np.asarray([o.log_joint for o in opts], dtype=float),
            hessian=[o.hessian for o in opts],
            covariance=[_full_covariance(o, model.n_parameters) for o in opts],
            diagnostics=[o.diagnostics for o in opts],
            free_mask=[o.free_mask for o in opts],
        ),
        profile=FitProfile(datetime=datetime.now().isoformat(timespec="seconds"), config=config, timings={**validated.timings}),
        model=model,
        data=copy.deepcopy(data),
    )
    if config.display:
        result.plot(subject=0, display=True)
    result.profile.timings["total"] = perf_counter() - fit_started
    return result
