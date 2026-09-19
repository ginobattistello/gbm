"""Numerical diagnostics for GBM Toolbox individual fits."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .state_model import prepare_subject_data


@dataclass
class ConvergenceSummary:
    """Whether the multi-start search agreed on a single solution.

    ``same_solution`` is the headline: it is true only when every start reached
    the same log joint *and* landed in the same place, so a high
    ``agreement_fraction`` with ``same_solution=False`` means the objective has
    a plateau or several comparable optima.
    """

    n_starts: int
    n_successful: int
    n_objective_agreeing: int
    agreement_fraction: float
    log_joint_spread: float
    max_parameter_distance: float
    same_solution: bool
    winning_success: bool
    winning_status: int | None
    winning_message: str
    gradient_norm: float


@dataclass
class HessianDiagnostics:
    """Spectrum of the observed Hessian and the posterior it implies.

    ``posterior_sd`` and ``posterior_correlation`` are ``None`` when the
    Hessian is not positive definite, i.e. when the Laplace approximation does
    not exist.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    posterior_sd: np.ndarray | None
    posterior_correlation: np.ndarray | None
    laplace_valid: bool
    laplace_fragile: bool


@dataclass
class DataInformationSpectrum:
    """How much the data adds to the prior, direction by direction.

    Eigenvalues are of the prior-preconditioned likelihood information, so they
    are unit-free: ``data_informed`` marks directions where the data carries
    more information than the prior. Negative values mean the objective curves
    the wrong way there.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    parameter_names: tuple[str, ...]
    negative_curvature: bool
    data_informed: np.ndarray


@dataclass
class LocalIdentifiability:
    """Rank of the prediction Jacobian at the MAP.

    A ``nullity`` above zero means some combination of parameters leaves every
    prediction unchanged, so the data cannot distinguish it;
    ``parameter_directions`` says which combination.
    """

    singular_values: np.ndarray
    rank: int
    nullity: int
    condition_number: float
    parameter_directions: np.ndarray
    parameter_names: tuple[str, ...]


@dataclass
class ObservationNoiseCorrection:
    """Observation-noise SD before and after correcting for parameter uncertainty.

    ``residual_sd`` is the SD implied by the MAP residuals alone, which treats
    the fitted parameters as known exactly.  ``corrected_sd`` additionally
    accounts for posterior uncertainty in those parameters via the expected
    residual energy under the Laplace posterior.

    ``inflation_factor`` is ``corrected_sd / residual_sd``; values meaningfully
    above 1 mean the model is flexible enough, relative to the number of
    trials, that the MAP residuals understate the true observation noise.
    """

    residual_sd: np.ndarray
    corrected_sd: np.ndarray
    reported_sd: np.ndarray
    uncertainty_trace: float
    n_trials: int
    n_free_parameters: int
    inflation_factor: np.ndarray
    valid: bool


@dataclass
class NoiseCorrelationSummary:
    """Posterior correlations involving the estimated noise parameters.

    Strong correlation between a noise SD and a model parameter means the two
    are trading off, and neither is pinned down on its own.
    """

    parameter_names: tuple[str, ...]
    correlation: np.ndarray | None


def convergence_diagnostics(fit, subject: int = 0, *, objective_tol=1e-4, parameter_tol=1e-3) -> ConvergenceSummary:
    """Compare the multi-start restarts for one subject.

    Two starts count as the same solution when their log joints agree to
    ``objective_tol`` and their endpoints to ``parameter_tol``, both relative
    to the scale of the quantity being compared.
    """
    diag = fit.math.diagnostics[subject]
    starts = list(diag.starts)
    if not starts:
        return ConvergenceSummary(
            n_starts=0,
            n_successful=1,
            n_objective_agreeing=1,
            agreement_fraction=1.0,
            log_joint_spread=0.0,
            max_parameter_distance=0.0,
            same_solution=True,
            winning_success=True,
            winning_status=diag.lbfgsb_status,
            winning_message=diag.lbfgsb_message,
            gradient_norm=diag.abs_grad,
        )
    vals = np.asarray([s.log_joint for s in starts], dtype=float)
    finite = np.isfinite(vals)
    best = np.max(vals[finite])
    agree = finite & (np.abs(vals - best) <= objective_tol * (1.0 + abs(best)))
    endpoints = np.asarray([s.final_parameters for s in starts], dtype=float)
    maxdist = 0.0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            maxdist = max(maxdist, float(np.linalg.norm(endpoints[i] - endpoints[j])))
    spread = float(np.max(vals[finite]) - np.min(vals[finite])) if np.any(finite) else np.inf
    same = bool(np.all(agree) and maxdist <= parameter_tol * (1.0 + np.linalg.norm(fit.output.parameters[subject])))
    return ConvergenceSummary(
        n_starts=len(starts),
        n_successful=sum(s.success for s in starts),
        n_objective_agreeing=int(np.sum(agree)),
        agreement_fraction=float(np.mean(agree)),
        log_joint_spread=spread,
        max_parameter_distance=maxdist,
        same_solution=same,
        winning_success=diag.lbfgsb_success,
        winning_status=diag.lbfgsb_status,
        winning_message=diag.lbfgsb_message,
        gradient_norm=diag.abs_grad,
    )


def posterior_hessian_diagnostics(fit, subject: int = 0) -> HessianDiagnostics:
    """Eigen-decompose the observed Hessian and report the Laplace posterior."""
    H = np.asarray(fit.math.hessian[subject], dtype=float)
    if H.size == 0:
        return HessianDiagnostics(
            eigenvalues=np.zeros(0),
            eigenvectors=np.zeros((0, 0)),
            condition_number=1.0,
            posterior_sd=np.zeros(0),
            posterior_correlation=np.zeros((0, 0)),
            laplace_valid=True,
            laplace_fragile=False,
        )
    vals, vecs = np.linalg.eigh(0.5 * (H + H.T))
    diag = fit.math.diagnostics[subject]
    cov_full = fit.math.covariance[subject]
    free = fit.math.free_mask[subject]
    if cov_full is None:
        sd = corr = None
    else:
        cov = cov_full[np.ix_(free, free)]
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        denom = np.outer(sd, sd)
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        np.fill_diagonal(corr, 1.0)
    return HessianDiagnostics(
        eigenvalues=vals,
        eigenvectors=vecs,
        condition_number=diag.hess_condition_number,
        posterior_sd=sd,
        posterior_correlation=corr,
        laplace_valid=diag.laplace_valid,
        laplace_fragile=diag.laplace_fragile,
    )


def prior_preconditioned_information(fit, subject: int = 0) -> DataInformationSpectrum:
    """Spectrum of the likelihood information measured in prior units.

    The likelihood information is recovered as ``H_post - prior precision`` and
    then whitened by the prior covariance, which makes its eigenvalues
    comparable across parameters on different scales: an eigenvalue above 1 is
    a direction the data constrains more tightly than the prior does.
    """
    free = fit.math.free_mask[subject]
    Hpost = np.asarray(fit.math.hessian[subject], dtype=float)
    Sigma0 = fit.input.prior_covariance[np.ix_(free, free)]
    if Hpost.size == 0:
        return DataInformationSpectrum(np.zeros(0), np.zeros((0, 0)), tuple(), False, np.zeros(0, dtype=bool))
    P0 = np.linalg.inv(Sigma0)
    Hlik = 0.5 * ((Hpost - P0) + (Hpost - P0).T)
    vals0, vecs0 = np.linalg.eigh(Sigma0)
    sqrtS = vecs0 @ np.diag(np.sqrt(vals0)) @ vecs0.T
    Htilde = 0.5 * (sqrtS @ Hlik @ sqrtS + (sqrtS @ Hlik @ sqrtS).T)
    vals, vecs = np.linalg.eigh(Htilde)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    names = tuple(np.asarray(fit.input.parameter_names, dtype=object)[free])
    return DataInformationSpectrum(
        eigenvalues=vals, eigenvectors=vecs, parameter_names=names, negative_curvature=bool(np.any(vals < 0.0)), data_informed=vals > 1.0
    )


def _free_prediction_function(fit, subject: int):
    """Build ``predictions(x_free) -> (T, d)`` and the pieces it needs.

    Several diagnostics differentiate the model's own trial-wise predictions
    with respect to the *free* parameters.  This returns that function together
    with the free-parameter indices and the MAP point in that reduced space, so
    each caller only has to say what it wants done with the Jacobian.

    The predictions come from ``model.evaluate_jax``, so differentiating them
    propagates through the latent trajectory exactly as the likelihood does and
    dynamical models need no special handling.
    """
    parameters = jnp.asarray(np.asarray(fit.output.parameters[subject], dtype=float))
    free = np.asarray(fit.math.free_mask[subject], dtype=bool)
    idx = np.flatnonzero(free)
    idx_j = jnp.asarray(idx, dtype=jnp.int32)
    prepared = prepare_subject_data(fit.data[subject])
    config = fit.profile.config

    def predictions(x_free):
        """Trial-wise predictions as ``(T, d)``, with fixed parameters held."""
        full = parameters.at[idx_j].set(x_free)
        out = fit.model.evaluate_jax(
            full,
            prepared,
            filter_max_iter=config.filter_max_iter,
            filter_tol=config.filter_tol,
            filter_damping=config.filter_damping,
            filter_jitter=config.filter_jitter,
        )
        prediction = out["prediction"]
        return prediction.reshape(prediction.shape[0], -1)

    return predictions, idx, parameters[idx_j]


def numerical_local_identifiability(fit, subject: int = 0, *, rank_tol=None) -> LocalIdentifiability:
    """Local prediction identifiability using an exact JAX Jacobian."""
    predictions, idx, x0 = _free_prediction_function(fit, subject)

    def flat_predictions(x_free):
        """Predictions flattened to one vector, for a single Jacobian."""
        p = predictions(x_free)
        if fit.model.family == "categorical":
            p = p[:, :-1]  # drop one class: the simplex makes it redundant
        return jnp.ravel(p)

    if len(idx) == 0:
        J = np.zeros((0, 0))
    else:
        J = np.asarray(jax.jacrev(flat_predictions)(x0), dtype=float)
    if J.size == 0:
        s = np.zeros(0)
        Vt = np.zeros((0, 0))
        rank = 0
    else:
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
        tol = rank_tol if rank_tol is not None else max(J.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        rank = int(np.sum(s > tol))
    cond = np.inf if s.size == 0 or rank < len(idx) or s[-1] == 0 else float(s[0] / s[-1])
    names = tuple(np.asarray(fit.input.parameter_names, dtype=object)[idx])
    return LocalIdentifiability(s, rank, len(idx) - rank, cond, Vt, names)


def noise_parameter_correlations(fit, subject: int = 0) -> NoiseCorrelationSummary:
    """Posterior correlation submatrix involving estimated Q/R log-SDs."""
    cov = fit.math.covariance[subject]
    if cov is None:
        return NoiseCorrelationSummary(tuple(), None)
    layout = fit.model.parameter_layout
    noise_idx = list(range(layout.process_noise_slice.start, layout.process_noise_slice.stop)) + list(
        range(layout.observation_noise_slice.start, layout.observation_noise_slice.stop)
    )
    if not noise_idx:
        return NoiseCorrelationSummary(tuple(), np.zeros((0, 0)))
    all_idx = list(range(cov.shape[0]))
    selected = sorted(set(all_idx + noise_idx))
    C = cov[np.ix_(selected, selected)]
    sd = np.sqrt(np.maximum(np.diag(C), 0.0))
    denom = np.outer(sd, sd)
    corr = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    names = tuple(np.asarray(fit.input.parameter_names, dtype=object)[selected])
    return NoiseCorrelationSummary(names, corr)


def observation_noise_correction(fit, subject: int = 0) -> ObservationNoiseCorrection:
    r"""Observation noise corrected for posterior parameter uncertainty.

    The MAP estimate of the observation noise is driven by the residuals at the
    fitted parameters, :math:`r_t = y_t - g_t(\hat\theta)`.  That treats
    :math:`\hat\theta` as known exactly, so a model flexible enough to absorb
    part of the noise into its own parameters leaves residuals that are too
    small and reports an observation SD that is too low.  With few trials, or
    many parameters, the effect is large: with five free parameters and 25
    trials the reported SD is biased low by roughly 10%.

    This diagnostic recomputes the noise from the *expected* residual energy
    under the Laplace posterior :math:`q(\theta) = N(\hat\theta, \Sigma)`.
    Linearising the prediction around the MAP,

    .. math::

        E_q\!\left[\sum_t (y_t - g_t(\theta))^2\right]
        \approx \sum_t \left(r_t^2 + J_t \Sigma J_t^\top\right),

    where :math:`J_t` is the Jacobian of the trial-``t`` prediction with
    respect to the free parameters.  The second term is the prediction
    uncertainty induced by parameter uncertainty; adding it is what turns the
    MLE-like estimate into the analogue of the unbiased :math:`SSE/(T-p)`.

    ``J`` is obtained by differentiating the model's own trajectory, so this
    works for dynamical models too: the derivative propagates through the
    latent states exactly as the likelihood does.

    This is a **diagnostic**: it reports what the noise would be under the
    correction, and changes neither the fit nor its log-evidence.  A large
    ``inflation_factor`` means the reported observation SD should not be taken
    at face value.

    The correction is evaluated once at the MAP, not iterated to
    self-consistency (the posterior covariance itself depends on the noise
    scale).  One step removes most of the bias; a small downward bias survives
    at the smallest trial counts.  See
    ``gbmtoolbox/dev/11_observation_noise_correction.py`` for the measured
    behaviour against the analytic result for a linear model.

    Only Gaussian observations have an estimated ``R``; other families return
    an empty, ``valid=False`` result.
    """
    model = fit.model
    if model.family != "gaussian":
        empty = np.zeros(0)
        return ObservationNoiseCorrection(empty, empty, empty, 0.0, 0, 0, empty, False)

    covariance = fit.math.covariance[subject]
    reported = np.asarray(fit.output.observation_noise_sd[subject], dtype=float)
    free = np.asarray(fit.math.free_mask[subject], dtype=bool)
    idx = np.flatnonzero(free)
    if covariance is None or idx.size == 0:
        # No valid Laplace posterior means no uncertainty to propagate.
        empty = np.zeros(0)
        return ObservationNoiseCorrection(empty, empty, reported, np.nan, 0, int(idx.size), empty, False)

    sigma = np.asarray(covariance, dtype=float)[np.ix_(idx, idx)]
    prediction_fn, _, x0 = _free_prediction_function(fit, subject)
    predictions = np.asarray(prediction_fn(x0), dtype=float)
    jacobian = np.asarray(jax.jacrev(prediction_fn)(x0), dtype=float)

    y = np.asarray(fit.data[subject]["y"], dtype=float).reshape(predictions.shape[0], -1)
    residual = y - predictions
    n_trials = residual.shape[0]

    # Per output dimension: sum_t r_t^2 and sum_t J_t Sigma J_t^T.
    sse = np.sum(residual**2, axis=0)
    trace = np.einsum("tdi,ij,tdj->d", jacobian, sigma, jacobian)

    residual_sd = np.sqrt(sse / n_trials)
    corrected_sd = np.sqrt((sse + trace) / n_trials)
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = np.divide(corrected_sd, residual_sd, out=np.ones_like(corrected_sd), where=residual_sd > 0)

    return ObservationNoiseCorrection(
        residual_sd=residual_sd,
        corrected_sd=corrected_sd,
        reported_sd=reported,
        uncertainty_trace=float(np.sum(trace)),
        n_trials=int(n_trials),
        n_free_parameters=int(idx.size),
        inflation_factor=inflation,
        valid=True,
    )
