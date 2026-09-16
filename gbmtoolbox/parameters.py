"""Resolved static-parameter layout for GBM Toolbox models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .priors import GaussianPrior, Priors, broadcast_prior


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
            raise ValueError("observation_noise_prior is required when observation_covariance='diagonal'")
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
