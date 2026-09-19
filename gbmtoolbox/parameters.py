"""Resolved static-parameter layout for GBM Toolbox models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, polygamma

from .priors import GaussianPrior, Priors, broadcast_prior

#: Prior sample size (``nu0``) of the default observation-noise prior, in
#: pseudo-observations.  ``nu0 = 2`` corresponds to the ``Ga(1, 1)`` prior on
#: observation precision used by default in the VBA toolbox.
DEFAULT_OBSERVATION_NOISE_PRIOR_SAMPLE_SIZE = 2.0

#: Prior mean and variance on ``log_observation_sd`` implied by that prior, by
#: moment matching (see ``observation_noise_prior_from_scale``).  They are
#: recorded as constants because they enter the Laplace evidence like any other
#: prior, so a change here changes every Gaussian log-evidence.
_DEFAULT_A0 = DEFAULT_OBSERVATION_NOISE_PRIOR_SAMPLE_SIZE / 2.0
DEFAULT_OBSERVATION_NOISE_PRIOR_MEAN = -0.5 * (float(digamma(_DEFAULT_A0)) - float(np.log(_DEFAULT_A0)))
DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE = 0.25 * float(polygamma(1, _DEFAULT_A0))


def _observation_noise_names(dim: int) -> tuple[str, ...]:
    if dim > 1:
        return tuple(f"log_observation_sd[{i}]" for i in range(dim))
    return ("log_observation_sd",)


def default_observation_noise_prior(dim: int) -> GaussianPrior:
    """Default prior on ``log_observation_sd``.

    This is the ``Ga(nu0/2, nu0/2)`` prior on observation precision with
    ``nu0 = 2`` -- i.e. VBA's default ``Ga(1, 1)`` -- moment-matched onto the
    log standard deviation that GBM Toolbox actually fits.  It places a 95%
    prior interval of roughly ``[0.38, 4.69]`` on the observation SD (matching
    moments does not preserve quantiles, so the exact ``Ga(1, 1)`` interval is
    ``[0.52, 6.29]``).

    It assumes an observation SD of order 1.  When the outcome is on a very
    different scale, state the scale instead with
    ``observation_noise_prior_from_scale``.
    """
    return GaussianPrior(
        [DEFAULT_OBSERVATION_NOISE_PRIOR_MEAN] * dim,
        [DEFAULT_OBSERVATION_NOISE_PRIOR_VARIANCE] * dim,
        names=_observation_noise_names(dim),
    )


def robust_scale(y) -> float:
    """Robust estimate of the scale of ``y`` (``1.4826 * MAD``).

    The constant makes this consistent with the standard deviation for Gaussian
    data, while a handful of outliers leave it essentially unchanged.  With 5%
    contamination the sample standard deviation of a unit-noise outcome inflates
    to roughly 3, whereas this stays near 1.  Returns ``0.0`` when the outcome
    carries no scale information (constant, or a single trial).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 2:
        return 0.0
    mad = float(np.median(np.abs(y - np.median(y))))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        # A median-based scale collapses when more than half the trials share a
        # value; fall back to the standard deviation before giving up.
        sd = float(np.std(y))
        scale = sd if np.isfinite(sd) and sd > 0.0 else 0.0
    return scale


def observation_noise_prior_from_scale(y=None, dim: int = 1, *, prior_scale: float | None = None, prior_sample_size: float = 2.0) -> GaussianPrior:
    r"""Observation-noise prior specified by a scale and a prior sample size.

    The noise prior is described the way a conjugate analysis would describe
    it -- by the noise scale you expect, ``s0``, and the number of
    pseudo-observations of prior information, ``nu0`` -- rather than by a
    variance on ``log_observation_sd``, which is hard to reason about.

    The equivalent conjugate prior on the observation *precision* is
    ``lambda ~ Gamma(a0, b0)`` with ``a0 = nu0/2`` and ``b0 = nu0*s0**2/2``,
    so that ``E[lambda] = 1/s0**2``.  GBM Toolbox parameterises noise by
    ``log_observation_sd``, so that prior is moment-matched onto this scale:
    ``log(sigma) = -log(lambda)/2`` has mean ``-(digamma(a0) - log(b0))/2`` and
    variance ``polygamma(1, a0)/4``.

    ``prior_sample_size`` sets the strength directly.  ``nu0 = 2`` (the default)
    gives a 95% prior interval of roughly ``[0.38, 4.69]`` times the scale --
    this is the ``Ga(1, 1)`` prior used by the VBA toolbox -- while
    ``nu0 = 10`` is substantially more informative.

    ``prior_scale`` is ``s0``.  Passing ``y`` instead estimates it robustly from
    the data as ``1.4826 * MAD(y)``, a conservative starting point that assumes
    the model explains none of the variance and that is insensitive to outliers.
    Reading the scale from the data is an empirical-Bayes choice, so it is
    opt-in rather than the default::

        model = StateModel(
            ...,
            observation_noise_prior=observation_noise_prior_from_scale(subject["y"]),
        )

    ves thBeing scale-based, the same model fitted to the same data expressed in
    different units gie same posterior over the parameters.  It removes
    the bias that comes from assuming an observation SD near 1; it does not make
    the noise better identified, which only more trials can do.
    """
    if prior_scale is None:
        if y is None:
            raise ValueError("pass either y or prior_scale")
        scale = robust_scale(y)
    else:
        scale = float(prior_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("prior_scale must be finite and positive")

    nu0 = float(prior_sample_size)
    if not np.isfinite(nu0) or nu0 <= 0.0:
        raise ValueError("prior_sample_size must be finite and positive")

    a0 = nu0 / 2.0
    variance = 0.25 * float(polygamma(1, a0))
    if scale <= 0.0:
        # No usable scale information: keep the shape of the prior but centre it
        # neutrally, exactly as the default prior does.
        centre = 0.0
    else:
        b0 = nu0 * scale**2 / 2.0
        centre = -0.5 * (float(digamma(a0)) - float(np.log(b0)))

    return GaussianPrior([centre] * dim, [variance] * dim, names=_observation_noise_names(dim))


@dataclass(frozen=True)
class ParameterLayout:
    """Bookkeeping for the full fitted vector ``[theta, phi, rho_Q, rho_R]``."""

    mean: np.ndarray
    covariance: np.ndarray
    names: tuple[str, ...]
    theta_slice: slice
    phi_slice: slice
    process_noise_slice: slice
    observation_noise_slice: slice
    process_noise_prior: GaussianPrior | None
    observation_noise_prior: GaussianPrior | None

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    @property
    def fixed_mask(self) -> np.ndarray:
        return np.isclose(np.diag(self.covariance), 0.0, atol=1e-14, rtol=0.0)

    @property
    def free_mask(self) -> np.ndarray:
        return ~self.fixed_mask

    def unpack(self, parameters):
        p = parameters
        return (p[self.theta_slice], p[self.phi_slice], p[self.process_noise_slice], p[self.observation_noise_slice])


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n), dtype=float)
    start = 0
    for b in blocks:
        d = b.shape[0]
        out[start : start + d, start : start + d] = b
        start += d
    return out


def build_parameter_layout(
    priors: Priors,
    *,
    process_noise_dim: int = 0,
    process_noise_prior: GaussianPrior | None = None,
    observation_noise_dim: int = 0,
    observation_noise_prior: GaussianPrior | None = None,
) -> ParameterLayout:
    """Construct the resolved full parameter vector and block-diagonal prior."""
    blocks = [priors.evolution, priors.observation]
    theta_slice = slice(0, priors.evolution.dim)
    phi_slice = slice(priors.evolution.dim, priors.dim)
    cursor = priors.dim

    pq = None
    if process_noise_dim:
        if process_noise_prior is None:
            raise ValueError("process_noise_prior is required when process_covariance='diagonal'")
        pq = broadcast_prior(process_noise_prior, process_noise_dim, prefix="log_process_sd")
        blocks.append(pq)
        process_slice = slice(cursor, cursor + pq.dim)
        cursor += pq.dim
    else:
        process_slice = slice(cursor, cursor)

    pr = None
    if observation_noise_dim:
        if observation_noise_prior is None:
            observation_noise_prior = default_observation_noise_prior(observation_noise_dim)
        pr = broadcast_prior(observation_noise_prior, observation_noise_dim, prefix="log_observation_sd")
        blocks.append(pr)
        observation_slice = slice(cursor, cursor + pr.dim)
        cursor += pr.dim
    else:
        observation_slice = slice(cursor, cursor)

    mean = np.concatenate([b.mean for b in blocks])
    covariance = _block_diag([b.covariance for b in blocks])
    names_list = list(priors.names)
    if pq is not None:
        names_list.extend(pq.names or tuple(f"log_process_sd[{i}]" for i in range(pq.dim)))
    if pr is not None:
        names_list.extend(pr.names or tuple(f"log_observation_sd[{i}]" for i in range(pr.dim)))
    names = tuple(names_list)
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique across all parameter blocks")

    return ParameterLayout(
        mean=mean,
        covariance=covariance,
        names=names,
        theta_slice=theta_slice,
        phi_slice=phi_slice,
        process_noise_slice=process_slice,
        observation_noise_slice=observation_slice,
        process_noise_prior=pq,
        observation_noise_prior=pr,
    )
