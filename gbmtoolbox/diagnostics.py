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
    prediction unchanged to first order; it may still change outcome variance;
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
    above 1 mean parameter-induced prediction uncertainty adds residual energy.
    They do not establish the true observation noise or estimator unbiasedness.
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
    are trading off, and neither is pinned down on its own. ``correlation`` has
    one row per free noise log-SD (``noise_names``) and one column per free
    parameter (``parameter_names``); it is ``None`` without a valid Laplace
    posterior.
    """

    noise_names: tuple[str, ...]
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
        # Every parameter is fixed: nothing was optimized, so there is nothing to disagree.
        return ConvergenceSummary(
            n_starts=0,
            n_successful=0,
            n_objective_agreeing=0,
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
    # optimize_map guarantees at least one finite start, so ``best`` exists.
    spread = float(best - np.min(vals[finite]))
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

    The likelihood information is recovered as ``H_post - local prior curvature`` and
    then whitened by the prior covariance, which makes its eigenvalues
    comparable across parameters on different scales: an eigenvalue above 1 is
    a direction with more local likelihood than prior curvature for Gaussian priors.
    For Gamma noise, this is instead a global-variance-scaled descriptive index.
    Gamma log-SD moment covariance supplies the scale, while its exact local
    curvature is subtracted to isolate the likelihood.
    """
    free = fit.math.free_mask[subject]
    Hpost = np.asarray(fit.math.hessian[subject], dtype=float)
    Sigma0 = fit.input.prior_covariance[np.ix_(free, free)]
    if Hpost.size == 0:
        return DataInformationSpectrum(np.zeros(0), np.zeros((0, 0)), tuple(), False, np.zeros(0, dtype=bool))
    P0 = np.linalg.inv(Sigma0)
    # Moment covariance defines the reporting scale. Subtract the actual
    # local prior curvature, not the Gaussian moment proxy, for Gamma noise.
    from .priors import GammaPrecisionPrior

    layout = fit.model.parameter_layout
    prior = layout.observation_noise_prior
    if isinstance(prior, GammaPrecisionPrior):
        full_idx = np.arange(layout.observation_noise_slice.start, layout.observation_noise_slice.stop)
        free_idx = np.flatnonzero(free)
        positions = np.searchsorted(free_idx, full_idx)
        rho = fit.output.parameters[subject, full_idx]
        P0[positions, positions] = 4.0 * prior.rate * np.exp(-2.0 * rho)
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
    """Local prediction sensitivity using an exact JAX Jacobian.

    The public name is retained for compatibility. Rank concerns the reported
    prediction, not the entire likelihood; covariance parameters can be
    identifiable even when this Jacobian has a null direction.
    """
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
        # Complete the right basis when outputs < parameters, without
        # allocating an outputs-by-outputs U for long datasets.
        _, s, Vt = np.linalg.svd(J, full_matrices=J.shape[0] < J.shape[1])
        tol = rank_tol if rank_tol is not None else max(J.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        rank = int(np.sum(s > tol))
    cond = np.inf if s.size == 0 or rank < len(idx) or s[-1] == 0 else float(s[0] / s[-1])
    names = tuple(np.asarray(fit.input.parameter_names, dtype=object)[idx])
    return LocalIdentifiability(s, rank, len(idx) - rank, cond, Vt, names)


def noise_parameter_correlations(fit, subject: int = 0) -> NoiseCorrelationSummary:
    """Posterior correlations between each estimated Q/R log-SD and every free parameter."""
    cov = fit.math.covariance[subject]
    if cov is None:
        return NoiseCorrelationSummary((), (), None)
    layout = fit.model.parameter_layout
    free = np.asarray(fit.math.free_mask[subject], dtype=bool)
    noise = np.zeros_like(free)
    noise[layout.process_noise_slice] = True
    noise[layout.observation_noise_slice] = True
    rows, cols = np.flatnonzero(noise & free), np.flatnonzero(free)
    names = np.asarray(fit.input.parameter_names, dtype=object)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(sd[rows], sd[cols])
    block = cov[np.ix_(rows, cols)]
    corr = np.divide(block, denom, out=np.zeros_like(block), where=denom > 0)
    return NoiseCorrelationSummary(tuple(names[rows]), tuple(names[cols]), corr)


def observation_noise_correction(fit, subject: int = 0) -> ObservationNoiseCorrection:
    r"""Linearized posterior residual-energy diagnostic for Gaussian outcomes.

    With residuals r_t and prediction Jacobians J_t at the MAP, compute
    residual_sd = sqrt(SSE/T) and corrected_sd = sqrt((SSE+U)/T), where
    U = sum_t J_t Sigma J_t.T. This approximates expected squared residuals
    under the static Laplace posterior; it is not a general unbiased noise
    estimator. For nonlinear predictions it omits prediction-curvature terms.

    The inflation factor compares these two residual scales, not the fitted
    reported_sd. Neither a factor near one nor a large factor certifies the
    fitted noise estimate. Priors, model mismatch and posterior approximation
    must be assessed separately. No fit or evidence is changed.

    The diagnostic requires deterministic Gaussian states and valid covariance
    with free parameters. It can also describe models with fixed observation
    covariance. It reports per-output aggregate residual scales, not a full
    covariance or heteroscedastic noise correction. Stochastic-state residuals
    also contain state uncertainty, so such models return valid=False.
    """
    model = fit.model
    if model.family != "gaussian" or model.has_state_uncertainty():
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
